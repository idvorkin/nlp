#!python3


import typer
from icecream import ic
from loguru import logger
from rich.console import Console
from elevenlabs.client import ElevenLabs
from elevenlabs import save, VoiceSettings
import sys
from pathlib import Path
from typing import Annotated, Optional
import random
import time
from pydantic import BaseModel
import subprocess
# from pydantic import BaseModel


console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def scratch():
    ic("hello world")


voices = {
    "fin": "fin",
    "igor": "Nvd5I2HGnOWHNU0ijNEy",
    "ammon": "AwdhqucUs1YyNaWbqQ57",
}
list_of_voices = ",".join(voices.keys())


@app.command()
def get_voices():
    client = ElevenLabs()
    voices = client.voices.get_all()
    ic(voices.voices)


@app.command()
def say(
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
    client = ElevenLabs()
    voice_settings = VoiceSettings(
        stability=0.4, similarity_boost=0.6, style=0.36, use_speaker_boost=True
    )

    audio = client.generate(
        text=to_speak,
        voice=voice,
        model=model,
        voice_settings=voice_settings,
    )
    ic(audio)

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
        save(audio, temp_path)
        outfile = temp_path

    if speak:
        ic(speak)
        # play via afplay
        subprocess.run(["afplay", outfile])
    if copy:
        import pbf

        pbf.copy(outfile)


# generated via [gpt.py2json](https://tinyurl.com/23dl535z)
class PodCast(BaseModel):
    class Item(BaseModel):
        Speaker: str
        ContentToSpeak: str


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
