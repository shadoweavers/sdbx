from __future__ import annotations

import copy
import heapq
import inspect
import logging
import sys
import threading
import traceback
import typing
from typing import List, Optional, Tuple, Union

import torch
from typing_extensions import TypedDict

from sdbx import interruption
from sdbx import model_management
from sdbx.component_model.abstract_prompt_queue import AbstractPromptQueue
from sdbx.component_model.executor_types import ExecutorToClientProgress
from sdbx.component_model.queue_types import QueueTuple, HistoryEntry, QueueItem, MAXIMUM_HISTORY_SIZE, ExecutionStatus
from sdbx.nodes.types import ExportedNodes

class Executor:
    def __init__(self, server: ExecutorToClientProgress):
        self.success = None
        self.server = server
        self.raise_exceptions = False
        self.reset()

    def reset(self):
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.status_messages = []
        self.success = True
        self.old_prompt = {}

    def add_message(self, event, data, broadcast: bool):
        self.status_messages.append((event, data))
        if self.server.client_id is not None or broadcast:
            self.server.send_sync(event, data, self.server.client_id)

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, interruption.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),

                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": error["current_outputs"],
            }
            self.add_message("execution_error", mes, broadcast=False)

        # Next, remove the subsequent outputs since they will not be executed
        to_delete = []
        for o in self.outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if o in self.old_prompt:
                    d = self.old_prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d

        if ex is not None and self.raise_exceptions:
            raise ex

    def execute(self, prompt, prompt_id, extra_data=None, execute_outputs: List[str] = None):
        with new_execution_context(ExecutionContext(self.server)):
            self._execute_inner(prompt, prompt_id, extra_data, execute_outputs)

    def _execute_inner(self, prompt, prompt_id, extra_data=None, execute_outputs: List[str] = None):
        if execute_outputs is None:
            execute_outputs = []
        if extra_data is None:
            extra_data = {}
        interruption.interrupt_current_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        self.status_messages = []
        self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            # delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d
            to_delete = []
            for o in self.object_storage:
                if o[0] not in prompt:
                    to_delete += [o]
                else:
                    p = prompt[o[0]]
                    if o[1] != p['class_type']:
                        to_delete += [o]
            for o in to_delete:
                d = self.object_storage.pop(o)
                del d

            for x in prompt:
                recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)

            current_outputs = set(self.outputs.keys())
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            model_management.cleanup_models(keep_clone_weights_loaded=True)
            self.add_message("execution_cached",
                             {"nodes": list(current_outputs), "prompt_id": prompt_id},
                             broadcast=False)
            executed = set()
            output_node_id = None
            to_execute = []

            for node_id in list(execute_outputs):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                # always execute the output that depends on the least amount of unexecuted nodes first
                memo = {}
                to_execute = sorted(list(
                    map(lambda a: (len(recursive_will_execute(prompt, self.outputs, a[-1], memo)), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]

                # This call shouldn't raise anything if there's an error deep in
                # the actual SD code, instead it will report the node where the
                # error was raised
                self.success, error, ex = recursive_execute(self.server, prompt, self.outputs, output_node_id,
                                                            extra_data, executed, prompt_id, self.outputs_ui,
                                                            self.object_storage)
                if self.success is not True:
                    self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)
                    break

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])
            self.server.last_node_id = None
            if model_management.DISABLE_SMART_MEMORY:
                model_management.unload_all_models()

    def _execute(self, prompt):
        