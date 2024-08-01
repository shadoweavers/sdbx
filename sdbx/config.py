import os
import ast
import enum
import shutil
import logging
import tomllib
import argparse
from functools import total_ordering, cached_property, lru_cache
from typing import List, Callable, Mapping, Any, Set, Iterator, Sequence, Dict, Union, Literal
from dataclasses import asdict, dataclass, field, fields
from watchdog.events import FileSystemEventHandler

from sdbx.component_model.files import get_package_as_path
from sdbx.nodes import NodeManager
from sdbx.utils import recursive_search

supported_pt_extensions = frozenset(['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl'])


def get_config_location():
    filename = "config.toml"

    if os.name == "nt":
        return os.path.join(os.environ['LOCALAPPDATA'], 'Shadowbox', filename)
    else:
        return os.path.join(os.path.expanduser('~'), '.config', 'shadowbox', filename)
    

@dataclass
class FolderPathsTuple:
    folder_name: str

    _last_update_time: int = 0

    paths: List[str] = field(default_factory=list)
    supported_extensions: Set[str] = field(default_factory=lambda: set(supported_pt_extensions))

    def __getitem__(self, item: Any):
        if item == 0:
            return self.paths
        if item == 1:
            return self.supported_extensions
        else:
            raise RuntimeError("unsupported tuple index")

    def __add__(self, other: "FolderPathsTuple"):
        assert self.folder_name == other.folder_name
        new_paths = list(frozenset(self.paths + other.paths))
        new_supported_extensions = self.supported_extensions | other.supported_extensions
        return FolderPathsTuple(self.folder_name, new_paths, new_supported_extensions)

    def __iter__(self) -> Iterator[Sequence[str]]:
        yield self.paths
        yield self.supported_extensions
    
    @property
    def folder_paths(self):
        return self.paths[:]
    
    @property
    @lru_cache(maxsize=None)
    def filename_list(self):
        if self._needs_filename_list_update():
            self.invalidate_cache()
        return self._get_filename_list()

    def invalidate_cache(self):
        self.__class__.filename_list.fget.cache_clear()
        self._last_update_time = os.path.getmtime(self) # possibly time.time()
    
    def _needs_filename_list_update(self):
        return os.path.getmtime(self) != self._last_update_time
        
    def _get_filename_list(self):
        def filter_files_extensions(files, extensions):
            return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions or len(extensions) == 0, files)))

        output_list = set()
        for x in self[0]:
            files = recursive_search(x, excluded_dir_names=[".git"])
            output_list.update(filter_files_extensions(files, self[1]))

        return sorted(list(output_list))

    def get_path_from_filename(self, filename):
        filename_split = os.path.split(filename)

        trusted_paths = []
        for folder in self.paths:
            folder_split = os.path.split(folder)
            abs_file_path = os.path.abspath(os.path.join(*folder_split, *filename_split))
            abs_folder_path = os.path.abspath(folder)
            if os.path.commonpath([abs_file_path, abs_folder_path]) == abs_folder_path:
                trusted_paths.append(abs_file_path)
            else:
                logging.error(f"attempted to access untrusted path {abs_file_path} in {self.folder_name} for filename {filename}")

        for trusted_path in trusted_paths:
            if os.path.isfile(trusted_path):
                return trusted_path

        return None


class FolderNames:
    def __init__(self, default_new_folder_path: str):
        self.contents: Dict[str, FolderPathsTuple] = dict()
        self.default_new_folder_path = default_new_folder_path

    def __getitem__(self, item) -> FolderPathsTuple:
        if not isinstance(item, str):
            raise RuntimeError("expected folder path")
        if item not in self.contents:
            default_path = os.path.join(self.default_new_folder_path, item)
            os.makedirs(default_path, exist_ok=True)
            self.contents[item] = FolderPathsTuple(item, paths=[default_path], supported_extensions=set())
        return self.contents[item]

    def __setitem__(self, key: str, value: FolderPathsTuple):
        assert isinstance(key, str)
        if isinstance(value, tuple):
            paths, supported_extensions = value
            value = FolderPathsTuple(key, paths, supported_extensions)
        self.contents[key] = value

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return iter(self.contents)

    def __delitem__(self, key):
        del self.contents[key]

    def items(self):
        return self.contents.items()

    def values(self):
        return self.contents.values()

    def keys(self):
        return self.contents.keys()


class LatentPreviewMethod(str, enum.Enum):
    NONE = "none"
    AUTO = "auto"
    LATENT2RGB = "latent2rgb"
    TAESD = "taesd"


@total_ordering
class VRAM(str, enum.Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    NONE = "none"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            order = [VRAM.NONE, VRAM.LOW, VRAM.NORMAL, VRAM.HIGH]
            return order.index(self) < order.index(other)
        return NotImplemented


class Precision(str, enum.Enum):
    MIXED = "mixed"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8E4M3FN = "float8_e4m3fn"
    FP8E5M2 = "float8_e5m2"


MixedPrecision = Literal[Precision.MIXED, Precision.FP32, Precision.FP16, Precision.BF16]
EncoderPrecision = Literal[Precision.FP32, Precision.FP16, Precision.BF16, Precision.FP8E4M3FN, Precision.FP8E5M2]


@dataclass
class ExtensionsConfig:
    disable: Union[bool, Literal['clients', 'nodes']] = False


@dataclass
class LocationConfig:
    clients: str = "clients"
    nodes: str = "nodes"
    input: str = "input"
    output: str = "output"
    models: str = "models"
    workflows: str = "workflows"


@dataclass
class WebConfig:
    listen: str = "127.0.0.1"
    port: str = "8188"
    external_address: str = field(default="localhost")
    enable_cors_header: Union[bool, str] = field(default=False)
    max_upload_size: int = field(default=100)
    max_queue_size: int = field(default=65536)
    auto_launch: bool = field(default=True)
    known_models: bool = field(default=True)
    preview_mode: LatentPreviewMethod = field(default=LatentPreviewMethod.AUTO)


@dataclass
class ComputationalConfig:
    gpu_only: bool = field(default=False)
    cuda_device: int = field(default=0)
    cuda_malloc: bool = field(default=True)
    cpu_only: bool = field(default=False)
    cpu_vae: bool = field(default=True)
    directml: Union[bool, int] = field(default=False)
    ipex_optimize: bool = field(default=False)
    xformers: bool = field(default=True)
    cross_attention: Literal['split', 'quad', 'torch'] = field(default="torch")
    upcast_attention: Union[bool, Literal['force']] = field(default=True)
    deterministic: bool = field(default=False)


@dataclass
class MemoryConfig:
    vram: VRAM = field(default=VRAM.NORMAL)
    smart_memory: bool = field(default=True)


@dataclass
class PrecisionConfig:
    fp: MixedPrecision = field(default=Precision.MIXED)
    unet: EncoderPrecision = field(default=Precision.FP32)
    vae: MixedPrecision = field(default=Precision.MIXED)
    text_encoder: EncoderPrecision = field(default=Precision.FP16)


@dataclass
class DistributedConfig:
    role: Literal[False, Literal['worker', 'frontend']] = field(default=False)
    name: str = field(default="shadowbox")
    connection_uri: str = field(default="amqp://guest:guest@127.0.0.1")


@dataclass
class OrganizationConfig:
    channels_first: bool = False


@dataclass
class Config:
    """
    Configuration options parsed from config.toml.
    """
    path: str = get_config_location()
    loglevel: Any = logging.INFO

    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    web: WebConfig = field(default_factory=WebConfig)
    computational: ComputationalConfig = field(default_factory=ComputationalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    organization: OrganizationConfig = field(default_factory=OrganizationConfig)

    def __post_init__(self):
        self.node_manager = NodeManager(self.get_path("nodes"))

    @classmethod
    def load(cls, filepath: str, loglevel=logging.INFO):
        path = os.path.dirname(filepath)

        try:
            with open(filepath, 'rb') as file:
                data = tomllib.load(file)

            config = Config.from_dict(
                data,
                path=path,
                loglevel=loglevel
            )
            
            return config
        except FileNotFoundError:
            logging.debug(f"Configuration file '{path}' not found. Generating config directory.")
            return Config.generate_new_config(path, loglevel)
        except tomllib.TOMLDecodeError as e:
            raise Exception(f"Error parsing TOML file: {e}")
        
    @classmethod
    def generate_new_config(cls, path: str, loglevel=logging.INFO):
        os.makedirs(path, exist_ok=True)

        config = cls(
            path=path, 
            loglevel=loglevel,
        )

        for subdir in config._path_dict.values():
            os.makedirs(subdir, exist_ok=True)
        
        shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.default.toml"), os.path.join(path, "config.toml"))
        open(os.path.join(path, "clients.toml"), 'a').close()
        open(os.path.join(path, "nodes.toml"), 'a').close()
        
        return config

    @classmethod
    def from_dict(cls, src, **kwargs):
        return cls._from_dict(cls, {**src, **kwargs})

    @staticmethod
    def _from_dict(cls: type, src: Mapping[str, Any]) -> Any:
        try:
            fieldtypes = {f.name: f.type for f in fields(cls)}
            src = { k.replace("-", "_"): v for k, v in src.items() }
            return cls(**{f: Config._from_dict(fieldtypes[f.replace("-", "_")], src[f]) for f in src})
        except TypeError as e:
            # logging.debug(type(e), e)
            return src # Not a dataclass field

    def rewrite(self, key, value):
        # rewrites the config.toml key to value
        pass

    def get_path(self, name):
        return self._path_dict[name]

    @cached_property
    def _path_dict(self):
        # root = {
        #     "clients": os.path.join(self.path, self.location.clients),
        #     "nodes": os.path.join(self.path, self.location.nodes),
        #     "input": os.path.join(self.path, self.location.input),
        #     "output": os.path.join(self.path, self.location.output),
        #     "models": os.path.join(self.path, self.location.models),
        #     "workflows": os.path.join(self.path, self.location.workflows)
        # }

        root = {
            f.name: os.path.join(self.path, getattr(self.location, f.name)) for f in fields(self.location)
        }

        for f in fields(self.location):
            if ".." in getattr(self.location, f.name):
                logging.error("Cannot set location outside of config path.")
                raise Exception()

        models = root["models"]

        models = {
            "models.checkpoints": os.path.join(models, "checkpoints"),
            "models.clip": os.path.join(models, "clip"),
            "models.clip_vision": os.path.join(models, "clip_vision"),
            "models.configs": os.path.join(models, "configs"),
            "models.controlnet": os.path.join(models, "controlnet"),
            "models.diffusers": os.path.join(models, "diffusers"),
            "models.embeddings": os.path.join(models, "embeddings"),
            "models.hypernetworks": os.path.join(models, "hypernetworks"),
            "models.huggingface": os.path.join(models, "huggingface"),
            "models.huggingface_cache": os.path.join(models, "huggingface_cache"),
            "models.loras": os.path.join(models, "loras"),
            "models.photomaker": os.path.join(models, "photomaker"),
            "models.style_models": os.path.join(models, "style_models"),
            "models.t2i_adapter": os.path.join(models, "t2i_adapter"),
            "models.unet": os.path.join(models, "unet"),
            "models.upscale_models": os.path.join(models, "upscale_models"),
            "models.vae": os.path.join(models, "vae"),
            "models.vae_approx": os.path.join(models, "vae_approx"),
        }

        return {**root, **models}

    @cached_property
    def folder_names(self):
        mfn = FolderNames() # Model Folder Names

        mfn["checkpoints"] = FolderPathsTuple("checkpoints", [self.get_path("models.checkpoints")], set(supported_pt_extensions))
        mfn["classifiers"] = FolderPathsTuple("classifiers", [self.get_path("models.classifiers")], {""})
        mfn["clip"] = FolderPathsTuple("clip", [self.get_path("models.clip")], set(supported_pt_extensions))
        mfn["clip_vision"] = FolderPathsTuple("clip_vision", [self.get_path("models.clip_vision")], set(supported_pt_extensions))
        mfn["configs"] = FolderPathsTuple("configs", [self.get_path("models.configs"), get_package_as_path("sdbx.configs")], {".yaml"})
        mfn["controlnet"] = FolderPathsTuple("controlnet", [self.get_path("models.controlnet"), self.get_path("models.t2i_adapter")], set(supported_pt_extensions))
        mfn["diffusers"] = FolderPathsTuple("diffusers", [self.get_path("models.diffusers")], {"folder"})
        mfn["embeddings"] = FolderPathsTuple("embeddings", [self.get_path("models.embeddings")], set(supported_pt_extensions))
        mfn["gligen"] = FolderPathsTuple("gligen", [self.get_path("models.gligen")], set(supported_pt_extensions))
        mfn["hypernetworks"] = FolderPathsTuple("hypernetworks", [self.get_path("models.hypernetworks")], set(supported_pt_extensions))
        mfn["huggingface"] = FolderPathsTuple("huggingface", [self.get_path("models.huggingface")], {""})
        mfn["huggingface_cache"] = FolderPathsTuple("huggingface_cache", [self.get_path("models.huggingface_cache")], {""})
        mfn["loras"] = FolderPathsTuple("loras", [self.get_path("models.loras")], set(supported_pt_extensions))
        mfn["photomaker"] = FolderPathsTuple("photomaker", [self.get_path("models.photomaker")], set(supported_pt_extensions))
        mfn["style_models"] = FolderPathsTuple("style_models", [self.get_path("models.style_models")], set(supported_pt_extensions))
        mfn["unet"] = FolderPathsTuple("unet", [self.get_path("models.unet")], set(supported_pt_extensions))
        mfn["vae"] = FolderPathsTuple("vae", [self.get_path("models.vae")], set(supported_pt_extensions))
        mfn["vae_approx"] = FolderPathsTuple("vae_approx", [self.get_path("models.vae_approx")], set(supported_pt_extensions))

        mfn["nodes"] = FolderPathsTuple("nodes", [os.path.join(self.path, "nodes")], set())

        return mfn

    def get_image_save_path(filename_prefix, output_dir, image_width=0, image_height=0):
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(input, image_width, image_height):
            input = input.replace("%width%", str(image_width))
            input = input.replace("%height%", str(image_height))
            return input

        filename_prefix = compute_vars(filename_prefix, image_width, image_height)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = str(os.path.join(output_dir, subfolder))

        if str(os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))) != str(output_dir):
            err = f"""**** ERROR: Saving image outside the output folder is not allowed.
                    full_output_folder: {os.path.abspath(full_output_folder)}
                            output_dir: {output_dir}
                            commonpath: {os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))}"""
            logging.error(err)
            raise Exception(err)

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1
        return full_output_folder, filename, counter, subfolder, filename_prefix


def parse() -> Config:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-c', '--config', type=str, default=get_config_location(), help='Location of the config file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('-s', '--silent', action='store_true', help='Silence all print to stdout.')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run in daemon mode (not associated with tty).')
    parser.add_argument('-h', '--help', action='help', help='See config.toml for more configuration options.')
    # parser.add_argument('--setup', action='store_true', help='Setup and exit.')

    args = parser.parse_args()

    loglevel = logging.INFO
    if args.verbose:
        loglevel = logging.DEBUG
    if args.silent:
        loglevel = logging.ERROR
    
    return Config.load(args.config, loglevel)

    # now give plugins a chance to add configuration
    # for entry_point in entry_points().select(group='sdbx.custom_config'):
    #     try:
    #         plugin_callable: ConfigurationExtender | ModuleType = entry_point.load()
    #         if isinstance(plugin_callable, ModuleType):
    #             # todo: find the configuration extender in the module
    #             plugin_callable = ...
    #         else:
    #             parser_result = plugin_callable(parser)
    #             if parser_result is not None:
    #                 parser = parser_result
    #     except Exception as exc:
    #         logging.error("Failed to load custom config plugin", exc_info=exc)

config = parse()