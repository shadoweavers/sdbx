import torch

import os
import json
import hashlib
import math
import random
import logging
from typing import Annotated as A

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
from huggingface_hub import hf_hub_download, snapshot_download
from natsort import natsorted
import numpy as np
import safetensors.torch

from sdbx import config
from sdbx.nodes.types import *

from .. import diffusers_load
from .. import samplers
from .. import sample
from .. import sd
from .. import utils
from .. import clip_vision as clip_vision_module
from .. import model_management

# from ..cmd import latent_preview
# from ..images import open_image
# from ..model_downloader import get_filename_list_with_downloadable, get_or_download, KNOWN_CHECKPOINTS, KNOWN_CLIP_VISION_MODELS, KNOWN_GLIGEN_MODELS, KNOWN_UNCLIP_CHECKPOINTS, KNOWN_LORAS, KNOWN_CONTROLNETS, KNOWN_DIFF_CONTROLNETS, KNOWN_VAES, KNOWN_APPROX_VAES, get_huggingface_repo_list, KNOWN_CLIP_MODELS, KNOWN_UNET_MODELS
# from .. import controlnet
# from ..open_exr import load_exr
# from .. import node_helpers
# from ..sd import VAE
# from ..utils import sdbx_tqdm


@node
def save_latent(
    samples: Latent, 
    filename_prefix: Annotated[str, Text()] = "sdbx", 
    prompt: Optional[str] = None, 
    extra_pnginfo: Optional[Dict[str, Any]] = None
) -> None:
    full_output_folder, filename, counter, subfolder, filename_prefix = config.get_image_save_path(filename_prefix, config.get_path("output"))

    # support save metadata for latent sharing
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {"prompt": prompt_info}
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata[x] = json.dumps(extra_pnginfo[x])

    file = f"{filename}_{counter:05}_.latent"

    results = [{
        "filename": file,
        "subfolder": subfolder,
        "type": "output"
    }]

    file = os.path.join(full_output_folder, file)

    output = {
        "latent_tensor": samples["samples"],
        "latent_format_version_0": torch.tensor([])
    }

    utils.save_torch_file(output, file, metadata=metadata)
    return {"ui": {"latents": results}}

@node
def load_latent(
    latent: Annotated[str, Literal[sorted(
        [f for f in os.listdir(config.get_path("inputs")) if os.path.isfile(os.path.join(config.get_path("inputs"), f)) and f.endswith(".latent")]
    )]]
) -> Latent:
    latent_path = folder_paths.get_annotated_filepath(latent)
    latent = safetensors.torch.load_file(latent_path, device="cpu")
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    samples = {"samples": latent["latent_tensor"].float() * multiplier}
    return samples
