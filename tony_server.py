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
    base = Path(f"/{modal_storage}")
    assistant_txt = '{"a":"b"}'
    # (base/"tony_assistant_spec.json").read_text()
    ic(assistant_txt)
    tony = json.loads(assistant_txt)
    tony_prompt = json.loads((base / "tony_system_prompt.md").read_text())
    # update system prompt
    tony["assistant"]["model"]["messages"][0]["content"] = tony_prompt
    ic(len(tony))
    return tony
