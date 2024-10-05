#!/usr/bin/env python3

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from loguru import logger
import openai
from pathlib import Path
import tempfile
import sounddevice as sd
import soundfile as sf

app = typer.Typer(no_args_is_help=True)
console = Console()

# Assuming you have set OPENAI_API_KEY as an environment variable
client = openai.AsyncOpenAI()


async def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    response = await client.audio.speech.create(model="tts-1", voice=voice, input=text)
    return response.content


async def transcribe_audio(audio_file: Path) -> str:
    with audio_file.open("rb") as file:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", file=file
        )
    return transcript.text


def play_audio(audio_data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_file_path = temp_file.name

    data, samplerate = sf.read(temp_file_path)
    sd.play(data, samplerate)
    sd.wait()

    Path(temp_file_path).unlink()


def record_audio(duration: int = 5) -> Path:
    console.print("Recording... Speak now!")
    samplerate = 44100
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, recording, samplerate)
        return Path(temp_file.name)


@app.command()
def speak(
    text: str,
    voice: str = typer.Option(
        "alloy", help="Voice to use (e.g., alloy, echo, fable, onyx, nova, shimmer)"
    ),
):
    """Convert text to speech and play it."""

    async def _speak():
        audio_data = await text_to_speech(text, voice)
        play_audio(audio_data)

    asyncio.run(_speak())


@app.command()
def listen(duration: int = typer.Option(5, help="Duration to record in seconds")):
    """Record audio and transcribe it."""

    async def _listen():
        audio_file = record_audio(duration)
        transcript = await transcribe_audio(audio_file)
        console.print(Panel(transcript, title="Transcription", expand=False))
        audio_file.unlink()

    asyncio.run(_listen())


@app.command()
def chat(
    duration: int = typer.Option(5, help="Duration to record each message in seconds"),
):
    """Have a voice conversation with the AI."""

    async def _chat():
        messages = []
        while True:
            # User's turn
            audio_file = record_audio(duration)
            user_message = await transcribe_audio(audio_file)
            audio_file.unlink()
            console.print(Panel(user_message, title="You", style="green", expand=False))
            messages.append({"role": "user", "content": user_message})

            # AI's turn
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages
            )
            ai_message = response.choices[0].message.content
            console.print(Panel(ai_message, title="AI", style="blue", expand=False))
            messages.append({"role": "assistant", "content": ai_message})

            # Convert AI's response to speech
            audio_data = await text_to_speech(ai_message)
            play_audio(audio_data)

            if "goodbye" in user_message.lower():
                break

    asyncio.run(_chat())


@logger.catch()
def main():
    app()


if __name__ == "__main__":
    main()
