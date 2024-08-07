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
def clip_text_encode(
    clip: CLIP, 
    text: A[str, Text(multiline=True, dynamic_prompts=True)]
) -> Conditioning:
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]

@node
def clip_set_last_layer(
    clip: CLIP,
    stop_at_clip_layer: Annotated[int, Numerical(min=-24, max=-1, step=1)] = -1
) -> CLIP:
    clip = clip.clone()
    clip.clip_layer(stop_at_clip_layer)
    return clip