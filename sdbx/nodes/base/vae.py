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


def vae_decode(vae: VAE, samples: Latent) -> Image:
    return vae.decode(samples["samples"])

def vae_decode_tiled(
    vae: VAE, 
    samples: Latent, 
    tile_size: Annotated[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Image:
    return vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8)

def vae_encode(vae: VAE, pixels: Image) -> Latent:
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}

def vae_encode_tiled(
    vae: VAE, 
    pixels: Image, 
    tile_size: Annotated[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Latent:
    t = vae.encode_tiled(pixels[:, :, :, :3], tile_x=tile_size, tile_y=tile_size)
    return {"samples": t}

@nodepath("inpaint/")
def vae_encode_for_inpaint(
    vae: VAE, 
    pixels: Image, 
    mask: Mask, 
    grow_mask_by: Annotated[int, Numerical(min=0, max=64, step=1)] = 6
) -> Latent:
    x = (pixels.shape[1] // vae.downscale_ratio) * vae.downscale_ratio
    y = (pixels.shape[2] // vae.downscale_ratio) * vae.downscale_ratio
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    pixels = pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % vae.downscale_ratio) // 2
        y_offset = (pixels.shape[2] % vae.downscale_ratio) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    # Grow mask by a few pixels to keep things seamless in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)
        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5
    t = vae.encode(pixels)

    return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}