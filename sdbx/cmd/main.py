import asyncio
import gc
import itertools
import logging
import os
import shutil
import threading
import time

from sdbx import config
from sdbx.cmd.server import PromptServer

from .extra_model_paths import load_extra_path_config
from .. import model_management
from ..cmd import cuda_malloc
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.queue_types import ExecutionStatus
from ..distributed.distributed_prompt_queue import DistributedPromptQueue
from ..distributed.server_stub import ServerStub


def prompt_worker(q: AbstractPromptQueue, _server: PromptServer):
    from ..cmd.execution import PromptExecutor

    e = PromptExecutor(_server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0
    current_time = 0.0
    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            _server.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.outputs_ui,
                        status=ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if _server.client_id is not None:
                _server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, _server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False


async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())


def cuda_malloc_warning():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run sdbx with: --disable-cuda-malloc\n")


async def main():
    # configure extra model paths earlier
    try:
        extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)
    except NameError:
        pass

    # if args.extra_model_paths_config:
    #     for config_path in itertools.chain(*args.extra_model_paths_config):
    #         load_extra_path_config(config_path)

    loop = asyncio.get_event_loop()
    server = await PromptServer(loop)
    if config.web.external_address != "localhost":
        server.external_address = config.web.external_address

    if config.distributed.role is not False:
        distributed = True
        q = DistributedPromptQueue(
            caller_server=server if config.distributed.role == "frontend" else None,
            connection_uri=config.distributed.connection_uri,
            is_caller=(config.distributed.role == "frontend"),
            is_callee=(config.distributed.role == "worker"),
            loop=loop,
            queue_name=config.distributed.name
        )
        await q.init()
    else:
        distributed = False
        from .execution import PromptQueue
        q = PromptQueue(server)
    server.prompt_queue = q

    server.add_routes()
    cuda_malloc_warning()

    # in a distributed setting, the default prompt worker will not be able to send execution events via the websocket
    worker_thread_server = server if not distributed else ServerStub()
    if not distributed or config.distributed.role == "worker":
        if distributed:
            logging.warning(f"Distributed workers started in the default thread loop cannot notify clients of progress updates. Instead of sdbx or main.py, use sdbx-worker.")
        threading.Thread(target=prompt_worker, daemon=True, args=(q, worker_thread_server,)).start()

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    config.folder_names["checkpoints"].paths.append(os.path.join(config.get_path("output"), "checkpoints"))
    config.folder_names["clip"].paths.append(os.path.join(config.get_path("output"), "clip"))
    config.folder_names["vae"].paths.append(os.path.join(config.get_path("output"), "vae"))

    call_on_start = None
    if config.web.auto_launch:
        def startup_server(address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0' or address == '':
                address = '127.0.0.1'
            webbrowser.open(f"http://{address}:{port}")

        call_on_start = startup_server

    # server.address = config.web.listen
    # server.port = config.web.port
    try:
        await run(server, address=config.web.listen, port=config.web.port, verbose=(config.loglevel is logging.DEBUG),
                  call_on_start=call_on_start)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.debug("\nStopped server")
    finally:
        if distributed:
            await q.close()


def entrypoint():
    asyncio.run(main())