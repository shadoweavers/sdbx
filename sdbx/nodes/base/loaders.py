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
from ... import sd
from .. import utils
from .. import clip_vision as clip_vision_module
from .. import model_management

# from ..cmd import latent_preview
# from ..images import open_image
from ...model_downloader import get_filename_list_with_downloadable, get_or_download, KNOWN_CHECKPOINTS, KNOWN_CLIP_VISION_MODELS, KNOWN_GLIGEN_MODELS, KNOWN_UNCLIP_CHECKPOINTS, KNOWN_LORAS, KNOWN_CONTROLNETS, KNOWN_DIFF_CONTROLNETS, KNOWN_VAES, KNOWN_APPROX_VAES, get_huggingface_repo_list, KNOWN_CLIP_MODELS, KNOWN_UNET_MODELS
from .. import controlnet
# from ..open_exr import load_exr
# from ..sd import VAE
# from ..utils import sdbx_tqdm

@nodepath("advanced/")
def checkpoint_loader(
    config_name: Annotated[str, Literal[config.folder_names["configs"].filename_list]],
    ckpt_name: Annotated[str, Literal[get_filename_list_with_downloadable("checkpoints", KNOWN_CHECKPOINTS)]]
) -> Tuple[Model, CLIP, VAE]:
    config_path = config.folder_names["configs"].get_path_from_filename(config_name)
    ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_CHECKPOINTS)
    return sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=config.folder_names["embeddings"].folder_paths)

def checkpoint_loader_simple(
    ckpt_name: Annotated[str, Literal[get_filename_list_with_downloadable("checkpoints", KNOWN_CHECKPOINTS)]]
) -> Tuple[Model, CLIP, VAE]:
    ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_CHECKPOINTS)
    out = sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=config.folder_names["embeddings"].folder_paths)
    return out[:3]

@nodepath("advanced/")
def diffusers_loader(
    model_path: Literal[
        tuple(frozenset(
            [os.path.relpath(root, start=search_path) for search_path in config.folder_names["diffusers"].folder_paths for root, _, files in os.walk(search_path, followlinks=True) if "model_index.json" in files] + get_huggingface_repo_list()
        ))
    ]
) -> Tuple[Model, CLIP, VAE]:
    for search_path in config.folder_names["diffusers"].folder_paths:
        if os.path.exists(search_path):
            path = os.path.join(search_path, model_path)
            if os.path.exists(path):
                model_path = path
                break
    if not os.path.exists(model_path):
        with sdbx_tqdm():
            model_path = snapshot_download(model_path)
    return diffusers_load.load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=config.folder_names["embeddings"].folder_paths)

def unclip_checkpoint_loader(
    ckpt_name: Annotated[str, Literal[get_filename_list_with_downloadable("checkpoints", KNOWN_UNCLIP_CHECKPOINTS)]]
) -> Tuple[Model, CLIP, VAE, CLIP_VISION]:
    ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_UNCLIP_CHECKPOINTS)
    out = sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=config.folder_names["embeddings"].folder_paths)
    return out

def load_lora(
    self,
    model: Model,
    clip: CLIP,
    lora_name: Annotated[str, Literal[get_filename_list_with_downloadable("loras", KNOWN_LORAS)]],
    strength_model: Annotated[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0,
    strength_clip: Annotated[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0
) -> Tuple[Model, CLIP]:
    if strength_model == 0 and strength_clip == 0:
        return model, clip

    lora_path = get_or_download("loras", lora_name, KNOWN_LORAS)
    lora = None
    if self.loaded_lora is not None:
        if self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            temp = self.loaded_lora
            self.loaded_lora = None
            del temp

    if lora is None:
        lora = utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)

    model_lora, clip_lora = sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    return model_lora, clip_lora

def lora_loader_model_only(
    model: Model,
    lora_name: Annotated[str, Literal[config.folder_names["loras"].filename_list]],
    strength_model: Annotated[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0
) -> Model:
    loader = LoraLoader()
    return loader.load_lora(model, None, lora_name, strength_model, 0)[0]

def vae_loader(
    vae_name: Annotated[str, Literal[VAELoader.vae_list()]]
) -> VAE:
    if vae_name in ["taesd", "taesdxl", "taesd3"]:
        sd_ = VAELoader.load_taesd(vae_name)
    else:
        vae_path = get_or_download("vae", vae_name, KNOWN_VAES)
        sd_ = utils.load_torch_file(vae_path)
    vae = sd.VAE(sd=sd_)
    return vae

def controlnet_loader(
    control_net_name: Annotated[str, Literal[get_filename_list_with_downloadable("controlnet", KNOWN_CONTROLNETS)]]
) -> CONTROL_NET:
    controlnet_path = get_or_download("controlnet", control_net_name, KNOWN_CONTROLNETS)
    controlnet_ = controlnet.load_controlnet(controlnet_path)
    return controlnet_

def diff_controlnet_loader(
    model: Model,
    control_net_name: Annotated[str, Literal[get_filename_list_with_downloadable("controlnet", KNOWN_DIFF_CONTROLNETS)]]
) -> CONTROL_NET:
    controlnet_path = get_or_download("controlnet", control_net_name, KNOWN_DIFF_CONTROLNETS)
    controlnet_ = controlnet.load_controlnet(controlnet_path, model)
    return controlnet_

def gligen_loader(gligen_name: str) -> Any:
    gligen_path = get_or_download("gligen", gligen_name, KNOWN_GLIGEN_MODELS)
    gligen = sd.load_gligen(gligen_path)
    return gligen

def unet_loader(unet_name: str) -> Any:
    unet_path = get_or_download("unet", unet_name, KNOWN_UNET_MODELS)
    model = sd.load_unet(unet_path)
    return model