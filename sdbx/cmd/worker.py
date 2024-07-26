import asyncio
import itertools
import os
import logging

from .extra_model_paths import load_extra_path_config
from .main_pre import args


async def main():
    # assume we are a worker
    args.distributed_queue_worker = True
    args.distributed_queue_frontend = False
    assert args.distributed_queue_connection_uri is not None, "Set the --distributed-queue-connection-uri argument to your RabbitMQ server"

    # configure paths
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        from ..cmd import folder_paths

        folder_paths.set_output_directory(output_dir)

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        from ..cmd import folder_paths

        folder_paths.set_input_directory(input_dir)

    if args.temp_directory:
        temp_dir = os.path.abspath(args.temp_directory)
        logging.info(f"Setting temp directory to: {temp_dir}")
        from ..cmd import folder_paths

        folder_paths.set_temp_directory(temp_dir)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    from ..distributed.distributed_prompt_worker import DistributedPromptWorker
    async with DistributedPromptWorker(connection_uri=args.distributed_queue_connection_uri,
                                       queue_name=args.distributed_queue_name):
        stop = asyncio.Event()
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
