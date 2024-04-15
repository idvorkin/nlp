#!python3


# import asyncio
from modal import Stub, web_endpoint
from typing import Dict
from icecream import ic
from pathlib import Path
import json

from modal import Image, Mount

default_image = Image.debian_slim(python_version="3.10").pip_install("icecream")

TONY_ASSISTANT_ID = "f5fe3b31-0ff6-4395-bc08-bc8ebbbf48a6"

stub = Stub("modal-tony-server")

modal_storage = "modal_readonly"


@stub.function(
    image=default_image,
    mounts=[Mount.from_local_dir(modal_storage, remote_path="/" + modal_storage)],
)
@web_endpoint(method="POST")
def assistant(input: Dict):
    ic(input)
    tony = json.loads(Path("/{modal_storage}/tony_soul.json").read_text())
    ic(tony)
    return tony
