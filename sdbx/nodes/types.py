from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Dict, Generic, Literal, get_type_hints, Union
from inspect import signature

from torch import Tensor
from torch.nn import Module as Model
from PIL import Image

from sdbx.sd import CLIP, VAE

# Primitives
# bool
# int
# str

# From primitives
Conditioning = Dict[str, Tensor] # dict of tensors?

# From existing classes
# CLIP
# Image
# Literal # choice
# Model
# VAE

# Annotated tensors
Latent = Annotated[Tensor, "latent"]
Mask = Annotated[Tensor, "mask"]

# Annotation classes
class Slider:
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float] = 1.0
    round: bool = False
    
    def __repr__(self):
        return f"Slider(min={self.min}, max={self.max}, step={self.step})"

class Numerical(Slider):
    def __repr__(self):
        return f"Numerical(min={self.min}, max={self.max}, step={self.step})"

class Text:
    multiline: bool = False
    dynamic_prompts: bool = False

    def __repr__(self):
        return f"Text(multiline={self.multiline}, dynamic_prompts={self.dynamic_prompts})"

class DependentInput:
    input_must_equal: Any
    new_inputs: dict[str, type]

class Name:
    name: str = ""

# Path decorator
def nodepath(path):
    def decorator(func):
        func.path = path
        return func
    return decorator