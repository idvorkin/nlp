#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "typer",
#     "loguru",
#     "rich",
#     "icecream",
#     "assemblyai",
#     "yt-dlp",
# ]
# ///

from datetime import datetime
import typer
import subprocess

from loguru import logger
from rich import print
from pathlib import Path
import assemblyai as aai
from icecream import ic
import os


app = typer.Typer(no_args_is_help=True)


def get_youtube_audio_url(youtube_url: str) -> str:
    """Extract audio URL from YouTube video using yt-dlp"""
    try:
        result = subprocess.run([
            "yt-dlp",
            "--get-url",
            "-f", "bestaudio",
            youtube_url
        ], capture_output=True, text=True, check=True)

        audio_url = result.stdout.strip()
        print(f"Extracted audio URL: {audio_url[:100]}...")
        return audio_url
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio URL: {e}")
        return None


@app.command()
def transcribe(
    path: Path = typer.Argument(None),
    url: str = typer.Option(None, help="YouTube URL to transcribe"),
    diarization: bool = True,
    srt: bool = False
):
    aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    if not aai.settings.api_key:
        print("Please set ASSEMBLYAI_API_KEY environment variable")
        print("Get your API key from: https://www.assemblyai.com/")
        return

    if url:
        # Handle YouTube URL
        if "youtube.com" in url or "youtu.be" in url:
            print("Extracting audio URL from YouTube...")
            audio_url = get_youtube_audio_url(url)
            if not audio_url:
                print("Failed to extract audio URL from YouTube")
                return
        else:
            # Direct audio URL
            audio_url = url
    elif path:
        # Handle local file
        audio_url = path.open("rb")
    else:
        # Default example
        audio_url = "https://github.com/AssemblyAI-Community/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

    config = aai.TranscriptionConfig(speaker_labels=diarization)

    ic(audio_url, datetime.now())
    transcript = aai.Transcriber().transcribe(audio_url, config)
    ic(transcript)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed: {transcript.error}")
        return

    if transcript.utterances:
        for utterance in transcript.utterances:
            print(f"Speaker {utterance.speaker}: {utterance.text}")
    else:
        print("No utterances found in transcript")
        print(f"Full transcript text: {transcript.text}")

    if srt:
        print(transcript.export_subtitles_srt())


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
