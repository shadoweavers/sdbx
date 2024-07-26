import asyncio
import logging
from asyncio import AbstractEventLoop
from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Optional

from aio_pika import connect_robust
from aio_pika.patterns import JsonRPC
from aiohttp import web
from aiormq import AMQPConnectionError

from .distributed_progress import DistributedExecutorToClientProgress
from .distributed_types import RpcRequest, RpcReply
from ..client.embedded_comfy_client import EmbeddedComfyClient
from ..cmd.main_pre import tracer
from ..component_model.queue_types import ExecutionStatus


class DistributedPromptWorker:
    """
    A distributed prompt worker.
    """

    def __init__(self, embedded_comfy_client: Optional[EmbeddedComfyClient] = None,
                 connection_uri: str = "amqp://localhost:5672/",
                 queue_name: str = "comfyui",
                 health_check_port: int = 9090,
                 loop: Optional[AbstractEventLoop] = None):
        self._rpc = None
        self._channel = None
        self._exit_stack = AsyncExitStack()
        self._queue_name = queue_name
        self._connection_uri = connection_uri
        self._loop = loop or asyncio.get_event_loop()
        self._embedded_comfy_client = embedded_comfy_client
        self._health_check_port = health_check_port
        self._health_check_site = None

    async def _health_check(self, request):
        return web.Response(text="OK", content_type="text/plain")

    async def _start_health_check_server(self):
        app = web.Application()
        app.router.add_get('/health', self._health_check)

        runner = web.AppRunner(app)
        await runner.setup()

        try:
            site = web.TCPSite(runner, port=self._health_check_port)
            await site.start()
            self._health_check_site = site
            logging.info(f"health check server started on port {self._health_check_port}")
        except OSError as e:
            if e.errno == 98:
                logging.warning(f"port {self._health_check_port} is already in use, health check disabled but starting anyway")
            else:
                logging.error(f"failed to start health check server with error {str(e)}, starting anyway")

    @tracer.start_as_current_span("Do Work Item")
    async def _do_work_item(self, request: dict) -> dict:
        await self.on_will_complete_work_item(request)
        try:
            request_obj = RpcRequest.from_dict(request)
        except Exception as e:
            request_dict_prompt_id_recovered = request["prompt_id"] \
                if request is not None and "prompt_id" in request else ""
            return asdict(RpcReply(request_dict_prompt_id_recovered, "", {},
                                   ExecutionStatus("error", False, [str(e)])))
        reply: RpcReply
        try:
            output_dict = await self._embedded_comfy_client.queue_prompt(request_obj.prompt,
                                                                         request_obj.prompt_id,
                                                                         client_id=request_obj.user_id)
            reply = RpcReply(request_obj.prompt_id, request_obj.user_token, output_dict,
                             ExecutionStatus("success", True, []))
        except Exception as e:
            reply = RpcReply(request_obj.prompt_id, request_obj.user_token, {},
                             ExecutionStatus("error", False, [str(e)]))

        await self.on_did_complete_work_item(request_obj, reply)
        return asdict(reply)

    @tracer.start_as_current_span("Initialize Prompt Worker")
    async def init(self):
        await self._exit_stack.__aenter__()
        try:
            self._connection = await connect_robust(self._connection_uri, loop=self._loop)
        except AMQPConnectionError as connection_error:
            logging.error(f"failed to connect to self._connection_uri={self._connection_uri}", connection_error)
            raise connection_error
        self._channel = await self._connection.channel()
        self._rpc = await JsonRPC.create(channel=self._channel, auto_delete=True, durable=False)

        if self._embedded_comfy_client is None:
            self._embedded_comfy_client = EmbeddedComfyClient(progress_handler=DistributedExecutorToClientProgress(self._rpc, self._queue_name, self._loop))
        if not self._embedded_comfy_client.is_running:
            await self._exit_stack.enter_async_context(self._embedded_comfy_client)

        await self._rpc.register(self._queue_name, self._do_work_item)
        await self._start_health_check_server()

    async def __aenter__(self) -> "DistributedPromptWorker":
        await self.init()
        return self

    async def _close(self):
        await self._rpc.close()
        await self._channel.close()
        await self._connection.close()
        if self._health_check_site:
            await self._health_check_site.stop()

    async def close(self):
        await self._close()
        await self._exit_stack.aclose()

    async def __aexit__(self, *args):
        await self._close()
        return await self._exit_stack.__aexit__(*args)

    async def on_did_complete_work_item(self, request: RpcRequest, reply: RpcReply):
        pass

    async def on_will_complete_work_item(self, request: dict):
        pass
