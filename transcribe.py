#!python3


from datetime import datetime
import typer

from loguru import logger
from rich import print
from pathlib import Path
import assemblyai as aai
from icecream import ic
import os


app = typer.Typer(no_args_is_help=True)


@app.command()
def transcribe(path: Path = typer.Argument(None), diarization=True):
    aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    default_audio_url = "https://github.com/AssemblyAI-Community/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"
    audio_url = path.open("rb") if path else default_audio_url

    config = aai.TranscriptionConfig(speaker_labels=diarization)

    ic(audio_url, datetime.now())
    transcript = aai.Transcriber().transcribe(audio_url, config)
    ic(transcript)

    for utterance in transcript.utterances:
        print(f"Speaker {utterance.speaker}: {utterance.text}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
