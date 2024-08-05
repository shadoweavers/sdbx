from sdbx import config

from . import clip
from . import conditioning
from . import latent
from . import loaders
from . import vae

def register_base_nodes():
    modules = [clip, conditioning, latent, loaders, vae]
    fns = [(module.__name__, fname, getattr(module, fname)) 
                        for module in modules 
                        for fname in dir(module) 
                        if callable(getattr(module, fname))]
    
    for module, fname, fn in fns:
        config.node_manager.register(fn, path=module)