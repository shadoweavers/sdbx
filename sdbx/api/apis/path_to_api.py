import typing
import typing_extensions

from sdbx.api.apis.paths.solidus import Solidus
from sdbx.api.apis.paths.api_v1_images_digest import ApiV1ImagesDigest
from sdbx.api.apis.paths.api_v1_prompts import ApiV1Prompts
from sdbx.api.apis.paths.embeddings import Embeddings
from sdbx.api.apis.paths.extensions import Extensions
from sdbx.api.apis.paths.history import History
from sdbx.api.apis.paths.interrupt import Interrupt
from sdbx.api.apis.paths.object_info import ObjectInfo
from sdbx.api.apis.paths.prompt import Prompt
from sdbx.api.apis.paths.queue import Queue
from sdbx.api.apis.paths.upload_image import UploadImage
from sdbx.api.apis.paths.view import View

PathToApi = typing.TypedDict(
    'PathToApi',
    {
    "/": typing.Type[Solidus],
    "/api/v1/images/{digest}": typing.Type[ApiV1ImagesDigest],
    "/api/v1/prompts": typing.Type[ApiV1Prompts],
    "/embeddings": typing.Type[Embeddings],
    "/extensions": typing.Type[Extensions],
    "/history": typing.Type[History],
    "/interrupt": typing.Type[Interrupt],
    "/object_info": typing.Type[ObjectInfo],
    "/prompt": typing.Type[Prompt],
    "/queue": typing.Type[Queue],
    "/upload/image": typing.Type[UploadImage],
    "/view": typing.Type[View],
    }
)

path_to_api = PathToApi(
    {
    "/": Solidus,
    "/api/v1/images/{digest}": ApiV1ImagesDigest,
    "/api/v1/prompts": ApiV1Prompts,
    "/embeddings": Embeddings,
    "/extensions": Extensions,
    "/history": History,
    "/interrupt": Interrupt,
    "/object_info": ObjectInfo,
    "/prompt": Prompt,
    "/queue": Queue,
    "/upload/image": UploadImage,
    "/view": View,
    }
)
