import pytest

from sdbx.api.components.schema.prompt import Prompt
from sdbx.config import Configuration
from sdbx.client.embedded_sdbx_client import EmbeddedComfyClient
from sdbx.model_downloader import add_known_models, KNOWN_LORAS
from sdbx.model_downloader_types import CivitFile

_workflows = {
    "lora_1": {
        "3": {
            "inputs": {
                "seed": 851616030078638,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": [
                    "10",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "negative": [
                    "7",
                    0
                ],
                "latent_image": [
                    "5",
                    0
                ]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": "masterpiece best quality girl",
                "clip": [
                    "10",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "7": {
            "inputs": {
                "text": "bad hands",
                "clip": [
                    "10",
                    1
                ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "8": {
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "4",
                    2
                ]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "sdbx",
                "images": [
                    "8",
                    0
                ]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        },
        "10": {
            "inputs": {
                "lora_name": "epi_noiseoffset2.safetensors",
                "strength_model": 1,
                "strength_clip": 1,
                "model": [
                    "4",
                    0
                ],
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "LoraLoader",
            "_meta": {
                "title": "Load LoRA"
            }
        }
    }
}


@pytest.fixture(scope="module", autouse=False)
@pytest.mark.asyncio
async def client(tmp_path_factory) -> EmbeddedComfyClient:
    config = Configuration()
    config.cwd = str(tmp_path_factory.mktemp("sdbx_test_cwd"))
    async with EmbeddedComfyClient(config) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow", _workflows.items())
async def test_workflow(workflow_name: str, workflow: dict, has_gpu: bool, client: EmbeddedComfyClient):
    if not has_gpu:
        pytest.skip("requires gpu")


    prompt = Prompt.validate(workflow)
    add_known_models("loras", KNOWN_LORAS, CivitFile(13941, 16576, "epi_noiseoffset2.safetensors"))
    # todo: add all the models we want to test a bit more elegantly
    outputs = await client.queue_prompt(prompt)

    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
