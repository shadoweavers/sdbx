import logging
import multiprocessing
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib
from typing import Tuple, List

import pytest
import requests

from sdbx.config import Configuration

# fixes issues with running the testcontainers rabbitmqcontainer on Windows
os.environ["TC_HOST"] = "localhost"


def get_lan_ip():
    """
    Finds the host's IP address on the LAN it's connected to.

    Returns:
        str: The IP address of the host on the LAN.
    """
    # Create a dummy socket
    s = None
    try:
        # Connect to a dummy address (Here, Google's public DNS server)
        # The actual connection is not made, but this allows finding out the LAN IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        if s is not None:
            s.close()
    return ip


def run_server(server_arguments: Configuration):
    from sdbx.cmd.main import main
    from sdbx.args import args
    import asyncio
    for arg, value in server_arguments.items():
        args[arg] = value
    asyncio.run(main())


@pytest.fixture(scope="function", autouse=False)
def has_gpu() -> bool:
    # mps
    has_gpu = False
    try:
        import torch
        has_gpu = torch.backends.mps.is_available() and torch.device("mps") is not None
        if has_gpu:
            from sdbx import model_management
            from sdbx.model_management import CPUState
            model_management.cpu_state = CPUState.MPS
    except ImportError:
        pass

    if not has_gpu:
        # ipex
        try:
            import intel_extension_for_pytorch as ipex
            has_gpu = ipex.xpu.device_count() > 0
        except ImportError:
            has_gpu = False

        if not has_gpu:
            # cuda
            try:
                import torch
                has_gpu = torch.device(torch.cuda.current_device()) is not None
            except:
                has_gpu = False

    if has_gpu:
        from sdbx import model_management
        from sdbx.model_management import CPUState
        if model_management.cpu_state != CPUState.MPS:
            model_management.cpu_state = CPUState.GPU if has_gpu else CPUState.CPU
    yield has_gpu


@pytest.fixture(scope="module", autouse=False)
def frontend_backend_worker_with_rabbitmq(tmp_path_factory) -> str:
    """
    starts a frontend and backend worker against a started rabbitmq, and yields the address of the frontend
    :return:
    """
    tmp_path = tmp_path_factory.mktemp("sdbx_background_server")
    processes_to_close: List[subprocess.Popen] = []
    from testcontainers.rabbitmq import RabbitMqContainer
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        frontend_command = [
            "sdbx",
            "--listen=0.0.0.0",
            "--port=9001",
            "--cpu",
            "--distributed-queue-frontend",
            f"-w={str(tmp_path)}",
            f"--distributed-queue-connection-uri={connection_uri}",
        ]

        processes_to_close.append(subprocess.Popen(frontend_command, stdout=sys.stdout, stderr=sys.stderr))
        backend_command = [
            "sdbx-worker",
            "--port=9002",
            f"-w={str(tmp_path)}",
            f"--distributed-queue-connection-uri={connection_uri}",
        ]

        processes_to_close.append(subprocess.Popen(backend_command, stdout=sys.stdout, stderr=sys.stderr))
        try:
            server_address = f"http://{get_lan_ip()}:9001"
            start_time = time.time()
            connected = False
            while time.time() - start_time < 60:
                try:
                    response = requests.get(server_address)
                    if response.status_code == 200:
                        connected = True
                        break
                except ConnectionRefusedError:
                    pass
                except Exception as exc:
                    logging.warning("", exc_info=exc)
                time.sleep(1)
            if not connected:
                raise RuntimeError("could not connect to frontend")
            yield server_address
        finally:
            for process in processes_to_close:
                process.terminate()


@pytest.fixture(scope="module", autouse=False)
def sdbx_background_server(tmp_path_factory) -> Tuple[Configuration, multiprocessing.Process]:
    tmp_path = tmp_path_factory.mktemp("sdbx_background_server")
    import torch
    # Start server

    configuration = Configuration()
    configuration.listen = "localhost"
    configuration.output_directory = str(tmp_path)
    configuration.input_directory = str(tmp_path)

    server_process = multiprocessing.Process(target=run_server, args=(configuration,))
    server_process.start()
    # wait for http url to be ready
    success = False
    for i in range(60):
        try:
            with urllib.request.urlopen(f"http://localhost:{configuration['port']}/object_info") as response:
                success = response.status == 200
                if success:
                    break
        except:
            pass
        time.sleep(1)
    if not success:
        raise Exception("Failed to start background server")
    yield configuration, server_process
    server_process.terminate()
    torch.cuda.empty_cache()


def pytest_collection_modifyitems(items):
    # Modifies items so tests run in the correct order

    LAST_TESTS = ['test_quality']

    # Move the last items to the end
    last_items = []
    for test_name in LAST_TESTS:
        for item in items.copy():
            print(item.module.__name__, item)
            if item.module.__name__ == test_name:
                last_items.append(item)
                items.remove(item)

    items.extend(last_items)


@pytest.fixture(scope="module")
def vae():
    from sdbx.nodes.base_nodes import VAELoader

    vae_file = "vae-ft-mse-840000-ema-pruned.safetensors"
    try:
        vae, = VAELoader().load_vae(vae_file)
    except FileNotFoundError:
        pytest.skip(f"{vae_file} not present on machine")
    return vae


@pytest.fixture(scope="module")
def clip():
    from sdbx.nodes.base_nodes import CheckpointLoaderSimple

    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[1]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")


@pytest.fixture(scope="module")
def model(clip):
    from sdbx.nodes.base_nodes import CheckpointLoaderSimple
    checkpoint = "v1-5-pruned-emaonly.safetensors"
    try:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint)[0]
    except FileNotFoundError:
        pytest.skip(f"{checkpoint} not present on machine")


@pytest.fixture(scope="function", autouse=False)
def use_temporary_output_directory(tmp_path: pathlib.Path):
    from sdbx.cmd import folder_paths

    orig_dir = folder_paths.get_output_directory()
    folder_paths.set_output_directory(tmp_path)
    yield tmp_path
    folder_paths.set_output_directory(orig_dir)


@pytest.fixture(scope="function", autouse=False)
def use_temporary_input_directory(tmp_path: pathlib.Path):
    from sdbx.cmd import folder_paths

    orig_dir = folder_paths.get_input_directory()
    folder_paths.set_input_directory(tmp_path)
    yield tmp_path
    folder_paths.set_input_directory(orig_dir)
