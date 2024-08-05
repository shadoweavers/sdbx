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
from sdbx.nodes import MAX_RESOLUTION
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
from .. import helpers
# from ..sd import VAE
# from ..utils import sdbx_tqdm


def conditioning_combine(a: Conditioning, b: Conditioning) -> Conditioning:
    return a + b

def conditioning_average(to: Conditioning, from_: Conditioning, to_strength: A[float, Slider(min=0, max=1, step=0.01)] = 1.0) -> Conditioning: 
    out: Conditioning = []

    if len(from_) > 1:
        logging.warning("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = from_[0][0]
    pooled_output_from = from_[0][1].get("pooled_output", None)

    for i in range(len(to)):
        t1 = to[i][0]
        pooled_output_to = to[i][1].get("pooled_output", pooled_output_from)
        t0 = cond_from[:,:t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

        tw = torch.mul(t1, to_strength) + torch.mul(t0, (1.0 - to_strength))
        t_to = to[i][1].copy()
        if pooled_output_from is not None and pooled_output_to is not None:
            t_to["pooled_output"] = torch.mul(pooled_output_to, to_strength) + torch.mul(pooled_output_from, (1.0 - to_strength))
        elif pooled_output_from is not None:
            t_to["pooled_output"] = pooled_output_from

        n = [tw, t_to]
        out.append(n)
    return out

def conditioning_concat(to: Conditioning, from_: Conditioning) -> Conditioning:
    out = []

    if len(from_) > 1:
        logging.warning("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = from_[0][0]

    for i in range(len(to)):
        t1 = to[i][0]
        tw = torch.cat((t1, cond_from),1)
        n = [tw, to[i][1].copy()]
        out.append(n)

    return out

def conditioning_set_area(
    conditioning: Conditioning, 
    width: A[int, Numerical(min=64, max=MAX_RESOLUTION, step=8)] = 64,
    height: A[int, Numerical(min=64, max=MAX_RESOLUTION, step=8)] = 64,
    x: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0.0,
    y: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0.0,
    strength: A[float, Numerical(min=0.0, max=10.0, step=0.01)] = 1.0
) -> Conditioning:
    c = helpers.conditioning_set_values(conditioning, {"area": ("percentage", height, width, y, x),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
    return c

def conditioning_set_area_percentage(
    conditioning: Conditioning, 
    width: A[float, Numerical(min=0.0, max=1.0, step=0.01)] = 1.0,
    height: A[float, Numerical(min=0.0, max=1.0, step=0.01)] = 1.0,
    x: A[float, Numerical(min=0.0, max=1.0, step=0.01)] = 0.0,
    y: A[float, Numerical(min=0.0, max=1.0, step=0.01)] = 0.0,
    strength: A[float, Numerical(min=0.0, max=10.0, step=0.01)] = 1.0
) -> Conditioning:
    c = helpers.conditioning_set_values(conditioning, {"area": ("percentage", height, width, y, x),
                                                                "strength": strength,
                                                                "set_area_to_bounds": False})
    return c

def conditioning_set_area_strength(conditioning: Conditioning, strength: A[float, Numerical(min=0.0, max=10.0, step=0.01)] = 1.0) -> Conditioning:
    c = helpers.conditioning_set_values(conditioning, {"strength": strength})
    return c

def conditioning_set_mask(
    conditioning: Conditioning, 
    mask: Mask,
    set_cond_area: Literal["default", "mask bounds"] = "default",
    strength: Annotated[float, Numerical(min=0.0, max=10.0, step=0.01)] = 1.0
) -> Conditioning:
    set_area_to_bounds = set_cond_area != "default"
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)

    c = helpers.conditioning_set_values(conditioning, {
        "mask": mask,
        "set_area_to_bounds": set_area_to_bounds,
        "mask_strength": strength
    })
    return c

def conditioning_zero_out(conditioning: Conditioning) -> Conditioning:
    c = []
    for t in conditioning:
        d = t[1].copy()
        if "pooled_output" in d:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    return c

def conditioning_set_timestep_range(
    conditioning: Conditioning,
    start: Annotated[float, Slider(min=0.0, max=1.0, step=0.001)] = 0.0,
    end: Annotated[float, Slider(min=0.0, max=1.0, step=0.001)] = 1.0
) -> Conditioning:
    c = helpers.conditioning_set_values(conditioning, {
        "start_percent": start,
        "end_percent": end
    })
    return c

def inpaint_model_conditioning(
    positive: Conditioning, 
    negative: Conditioning,
    vae: VAE,
    pixels: Image,
    mask: Mask
) -> Tuple[Conditioning, Conditioning, Latent]:
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5
    concat_latent = vae.encode(pixels)
    orig_latent = vae.encode(orig_pixels)

    out_latent = {
        "samples": orig_latent,
        "noise_mask": mask
    }

    out = []
    for conditioning in [positive, negative]:
        c = helpers.conditioning_set_values(conditioning, {
            "concat_latent_image": concat_latent,
            "concat_mask": mask
        })
        out.append(c)
    return out[0], out[1], out_latent

## Controlnet

def controlnet_apply(
    conditioning: Conditioning,
    control_net: Any,
    image: A[Tensor, "image"],
    strength: A[float, Slider(min=0.0, max=10.0, step=0.01)] = 1.0
) -> Conditioning:
    if strength == 0:
        return conditioning

    c = []
    control_hint = image.movedim(-1, 1)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = control_net.copy().set_cond_hint(control_hint, strength)
        if 'control' in t[1]:
            c_net.set_previous_controlnet(t[1]['control'])
        n[1]['control'] = c_net
        n[1]['control_apply_to_uncond'] = True
        c.append(n)
    return c

def controlnet_apply_advanced(
    positive: Conditioning,
    negative: Conditioning,
    control_net: Any,
    image: A[Tensor, "image"],
    strength: A[float, Slider(min=0.0, max=10.0, step=0.01)] = 1.0,
    start_percent: A[float, Slider(min=0.0, max=1.0, step=0.001)] = 0.0,
    end_percent: A[float, Slider(min=0.0, max=1.0, step=0.001)] = 1.0,
    vae: VAE = None
) -> Tuple[Conditioning, Conditioning]:
    if strength == 0:
        return positive, negative

    control_hint = image.movedim(-1, 1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()
            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return out[0], out[1]