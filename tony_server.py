#!python3


# import asyncio
from modal import Stub, web_endpoint
from typing import Dict
from icecream import ic
from pathlib import Path
import json
import httpx
import datetime

from modal import Image, Mount

default_image = Image.debian_slim(python_version="3.10").pip_install(
    ["icecream", "httpx"]
)

TONY_ASSISTANT_ID = "f5fe3b31-0ff6-4395-bc08-bc8ebbbf48a6"

stub = Stub("modal-tony-server")

modal_storage = "modal_readonly"


def get_weather(line):
    desc = line["weatherDesc"][0]["value"]
    degree = line["FeelsLikeF"]
    return degree, desc


def parse_weather(weather):
    current = weather["current_condition"][0]
    degree, desc = get_weather(current)
    output = f"""Current Weather: {desc}: degrees {degree}
Weather Throught the Day hour, condition, temperature"""

    hourly = weather["weather"][0]["hourly"]
    for hour in hourly:
        time = hour["time"]
        degree, desc = get_weather(hour)
        output += f" {time}: {desc}: degrees {degree} \n"

    return output


def weather_from_server():
    params = {"format": "j1"}
    response = httpx.get("https://wttr.in/seattle?format=json", params=params)
    return response.json()


@stub.function(
    image=default_image,
    mounts=[Mount.from_local_dir(modal_storage, remote_path="/" + modal_storage)],
)
@web_endpoint(method="POST")
def assistant(input: Dict):
    ic(input)
    base = Path(f"/{modal_storage}")
    assistant_txt = (base / "tony_assistant_spec.json").read_text()
    tony = json.loads(assistant_txt)
    tony_prompt = (base / "tony_system_prompt.md").read_text()
    # Add context to system prompt
    extra_state = f"""
# Current state
date in UTC (but convert to PST as that's where Igor is, though don't mention it) :{datetime.datetime.now()} -
weather:
{parse_weather(weather_from_server())}
    """
    tony_prompt += extra_state
    ic(extra_state)
    # update system prompt
    tony["assistant"]["model"]["messages"][0]["content"] = tony_prompt
    ic(len(tony))
    return tony
