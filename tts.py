#!python3


import typer
from icecream import ic
from loguru import logger
from rich.console import Console
from elevenlabs import generate, play, set_api_key
import json
import sys
from pathlib import Path
from typing import Annotated, Optional
import random
import time
# from pydantic import BaseModel


console = Console()
app = typer.Typer(no_args_is_help=True)


def setup_secret():
    PASSWORD = "replaced_from_secret_box"
    secret_file = Path.home() / "gits/igor2/secretBox.json"

    SECRETS = json.loads(secret_file.read_text())
    PASSWORD = SECRETS["elevenlabs_api_key"]

    return PASSWORD


api_key = setup_secret()
set_api_key(api_key)


@app.command()
def scratch():
    ic("hello world")


voices = {
    "fin": "fin",
    "igor": "i55Y1MKKlAAPtIu2H5su",
    "ammon": "AwdhqucUs1YyNaWbqQ57",
}
list_of_voices = ",".join(voices.keys())


@app.command()
def say(
    multilingual: bool = True,
    voice: Annotated[
        str, typer.Option(help=f"Model any of: {list_of_voices}")
    ] = "igor",
    fast: bool = True,
    copy: bool = False,
    outfile: Optional[Path] = None,
    speak: bool = True,
):
    # record how long it takes
    start = time.time()
    to_speak = "\n".join(sys.stdin.readlines())
    model = "eleven_turbo_v2" if fast else "eleven_multilingual_v2"
    voice = voices[voice]
    ic(voice, model)
    audio = generate(
        text=to_speak,
        voice=voice,
        model=model,
    )

    print(f"Took {round(time.time() -start,3)} seconds")
    if outfile is not None:
        # write to it
        outfile.write_bytes(audio)
        print(outfile)
    else:
        temp_path = Path.home() / "tmp/tts" / f"{random.random()}.mp3"
        # make the dir if it doesn't exist
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        print(temp_path)
        temp_path.write_bytes(audio)

    if speak:
        play(audio)
    if copy:
        import pbf

        pbf.copy(temp_path)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
