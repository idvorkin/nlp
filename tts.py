#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "elevenlabs",
#     "icecream",
#     "loguru",
#     "pydantic",
#     "rich",
#     "pbf",
#     "google-cloud-texttospeech",
#     "pydub",
# ]
# ///


import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Iterator, Optional
import asyncio

import typer
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from icecream import ic
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
import os
import re

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def scratch():
    ic("hello world")


voices = {
    "fin": "fin",
    "igor": "Nvd5I2HGnOWHNU0ijNEy",
    "ammon": "AwdhqucUs1YyNaWbqQ57",
    "rachel": "VrNQNREmlwaHD01224L3",
}
list_of_voices = ",".join(voices.keys())


@app.command()
def list_voices():
    client = ElevenLabs()
    voices = client.voices.get_all()
    for voice in voices:
        ic(voice)


def generate_audio(
    text: str,
    voice: str,
    voice_settings: VoiceSettings = VoiceSettings(
        stability=0.4, similarity_boost=0.6, style=0.36, use_speaker_boost=True
    ),
    model: str = "eleven_turbo_v2",
) -> Iterator[bytes]:
    client = ElevenLabs()
    voice = voices[voice]
    return client.generate(
        text=text,
        voice=voice,
        model=model,
        voice_settings=voice_settings,
    )


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
    # look up voice in voices
    voice = voices[voice]
    # record how long it takes
    start = time.time()
    to_speak = "\n".join(sys.stdin.readlines())
    model = "eleven_turbo_v2" if fast else "eleven_multilingual_v2"
    ic(voice, model)
    api_key = os.getenv("ELEVEN_API_KEY")

    client = ElevenLabs(api_key=api_key)

    voice_settings = VoiceSettings(
        stability=0.4, similarity_boost=0.6, style=0.36, use_speaker_boost=True
    )

    audio = client.generate(
        text=to_speak,
        voice=voice,
        model=model,
        voice_settings=voice_settings,
    )
    # unwrapp the iterator
    audio = b"".join(audio)

    print(f"Took {round(time.time() - start, 3)} seconds")
    if outfile is None:
        temp_path = Path.home() / "tmp/tts" / f"{random.random()}.mp3"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        outfile = temp_path

    outfile.write_bytes(audio)
    print(outfile)
    if speak:
        ic(speak)
        # play via afplay
        subprocess.run(["afplay", outfile])
    if copy:
        import pbf

        pbf.copy(outfile)


@app.command()
def podcast(
    infile: Path = Path("podcast.json"),
    outdir: Optional[Path] = None,
    speak: bool = True,
):
    # create output dir name of podcast_<infile>, remove extension
    # if it exists throw
    if outdir is None:
        outdir = Path(f"podcast_{infile.stem}")
    else:
        outdir = Path(outdir)
    # throw if it exists
    if outdir.exists():
        pass
        # raise ValueError(f"Output directory {outdir} already exists")
    outdir.mkdir(parents=True, exist_ok=True)

    # inffile is a json array of PodcastItems, load it up into python
    items = []
    with open(infile, "r") as f:
        json_items = json.load(f)
        items = [PodCastItem.model_validate(item) for item in json_items]
        ic(items)

    for index, item in enumerate(items, start=1):
        # create a temp path
        temp_path = outdir / f"{item.Speaker}_{index:03d}.mp3"
        ic(temp_path)
        # if it exists throw
        if temp_path.exists():
            ic(f"Output file {temp_path} already exists - skipping")
            continue
        else:
            # write out the audio to the file
            voice_label = ""
            if item.Speaker == "Host":
                voice_label = "igor"
            elif item.Speaker == "Guest":
                voice_label = "rachel"
            else:
                raise ValueError(f"Unknown speaker {item.Speaker}")

            audio = generate_audio(item.ContentToSpeak, voice_label)
            with open(temp_path, "wb") as f:
                audio = b"".join(audio)
                f.write(audio)


@app.command()
def google_multi(pod=Path("pod.json"), speak: bool = True):
    from google.cloud import texttospeech_v1beta1 as tts
    from google.cloud.texttospeech_v1beta1 import (
        MultiSpeakerMarkup,
        AudioEncoding,
        VoiceSelectionParams,
        SynthesisInput,
        AudioConfig,
    )

    conversation = []
    # load the podcast
    with open(pod, "r") as f:
        podcast = json.load(f)
        conversation = podcast["conversation"]

    # Define the conversation as a list of tuples (speaker, text)

    markupTurns = [
        MultiSpeakerMarkup.Turn(text=turn["text"], speaker=turn["speaker"])
        for turn in conversation
    ]

    # Remap speakers to be R,S,M be dynamic in how you build that
    original_speakers = set([turn.speaker for turn in markupTurns])
    ic(original_speakers)
    valid_google_speakers = "R,S,T,U".split(",")
    # map from original speakers to valid google speakers
    speaker_map = {
        speaker: valid_google_speakers[index]
        for index, speaker in enumerate(original_speakers)
    }
    ic(speaker_map)
    for turn in markupTurns:
        turn.speaker = speaker_map[turn.speaker]

    multi_speaker_markup = MultiSpeakerMarkup(turns=markupTurns)
    ic(multi_speaker_markup)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = tts.TextToSpeechClient().synthesize_speech(
        input=SynthesisInput(multi_speaker_markup=multi_speaker_markup),
        voice=VoiceSelectionParams(
            language_code="en-US", name="en-US-Studio-MultiSpeaker"
        ),
        audio_config=AudioConfig(audio_encoding=AudioEncoding.MP3),
    )

    # The response's audio_content is binary.
    output_path = "pod.wav"  # not sure why, but it's only outputing wav
    ic(output_path)
    with open(output_path, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

    if speak:
        ic(speak)
        # play via afplay
        subprocess.run(["afplay", output_path])


@app.command()
def merge_audio(directory: Path):
    from pydub import AudioSegment
    # Specify the directory where youjjjjr audio files are located

    # Function to extract the numeric part from the filename for sorting
    def extract_number(file_name):
        return int(re.search(r"\d+", file_name).group())

    # Get all the files in the directory that match the pattern
    files = [f for f in os.listdir(directory) if f.endswith(".mp3")]

    # Sort files by the numeric part extracted from the filenames
    files.sort(key=extract_number)

    # Initialize an empty AudioSegment object
    combined = AudioSegment.empty()

    # Loop through the files and merge them
    for file in files:
        audio = AudioSegment.from_mp3(os.path.join(directory, file))
        combined += audio

    # Export the merged audio file
    output_path = os.path.join(directory, "merged_audio.mp3")
    combined.export(output_path, format="mp3")

    print(f"Merged audio saved to {output_path}")


# generated via [gpt.py2json](https://tinyurl.com/23dl535z)
class PodCastItem(BaseModel):
    Speaker: str
    ContentToSpeak: str


@logger.catch()
def app_wrap_loguru():
    app()


async def generate_single_voice(turn, i, temp_dir, speed):
    """Generate a single voice segment asynchronously"""
    from google.cloud import texttospeech as tts
    from google.cloud.texttospeech import (
        AudioEncoding,
        VoiceSelectionParams,
        SynthesisInput,
        AudioConfig,
    )

    client = tts.TextToSpeechAsyncClient()
    speaker_output = temp_dir / f"dialog_chirp3hd_speaker_{i + 1}.mp3"

    # Use different Chirp 3: HD voices for different speakers
    voice_name = (
        "en-US-Chirp3-HD-Puck" if turn["speaker"] == "Alex" else "en-US-Chirp3-HD-Leda"
    )

    try:
        # Use text input (not markup) for Chirp 3: HD voices
        synthesis_input = SynthesisInput(text=turn["text"])

        # Apply speed control using speaking_rate in AudioConfig
        audio_config = AudioConfig(
            audio_encoding=AudioEncoding.MP3,
            speaking_rate=speed,  # This is the correct way for Chirp 3: HD
        )

        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=VoiceSelectionParams(language_code="en-US", name=voice_name),
            audio_config=audio_config,
        )

        with open(speaker_output, "wb") as out:
            out.write(response.audio_content)

        print(f"   Generated: {speaker_output}")
        return speaker_output

    except Exception as e:
        print(f"   ❌ Failed to generate {speaker_output}: {e}")
        return None


@app.command()
def compare_voices(speak: bool = True, speed: float = 1.5):
    """Generate dialog with Chirp 3: HD voices with adjustable speed (parallel processing)

    Documentation: https://cloud.google.com/text-to-speech/docs/chirp3-hd
    """

    # Create a short dialog between two people
    dialog = [
        {
            "speaker": "Alex",
            "text": "Hey Sarah, have you tried the new AI voice synthesis technology?",
        },
        {
            "speaker": "Sarah",
            "text": "Not yet! I've heard it's incredibly realistic. What's your experience been like?",
        },
        {
            "speaker": "Alex",
            "text": "It's amazing! The voices sound so natural, and you can even have multiple speakers in one conversation.",
        },
        {
            "speaker": "Sarah",
            "text": "That's fascinating! I wonder how it compares to traditional text-to-speech systems.",
        },
        {
            "speaker": "Alex",
            "text": "The difference is night and day. The intonation, emotion, and naturalness are remarkable.",
        },
        {
            "speaker": "Sarah",
            "text": "I'll definitely have to give it a try. Thanks for the recommendation!",
        },
    ]

    # Create temp directory for output files
    temp_dir = Path.home() / "tmp/tts/voice_comparison"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("🎭 Creating dialog with Chirp 3: HD voices (parallel processing)...")
    print(f"Speed: {speed}x")
    print("\nDialog content:")
    for turn in dialog:
        print(f"  {turn['speaker']}: {turn['text']}")

    print(f"\n📁 Output directory: {temp_dir}")

    async def generate_all_voices():
        """Generate all voices concurrently with semaphore limiting"""
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

        async def generate_with_semaphore(turn, i):
            async with semaphore:
                return await generate_single_voice(turn, i, temp_dir, speed)

        # Create tasks for all voice generations
        tasks = [generate_with_semaphore(turn, i) for i, turn in enumerate(dialog)]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    # Generate with Chirp 3: HD voices
    print("\n🚀 Generating with Chirp 3: HD voices (parallel)...")
    start_time = time.time()

    # Run the async function
    chirp_outputs = asyncio.run(generate_all_voices())

    chirp_time = time.time() - start_time
    print(f"✅ Chirp 3: HD voices completed in {chirp_time:.2f} seconds")

    # Play the results
    if speak:
        print("\n🔊 Playing Chirp 3: HD voices...")
        if len(chirp_outputs) == 1:
            print("Playing single file...")
            subprocess.run(["afplay", chirp_outputs[0]])
        else:
            print("Playing individual voice segments...")
            for output_file in chirp_outputs:
                if Path(output_file).exists():
                    subprocess.run(["afplay", output_file])
                    time.sleep(0.5)  # Small pause between segments

    print("\n📊 Generation Summary:")
    print(f"   Chirp 3: HD Voices: {chirp_time:.2f}s - {len(chirp_outputs)} files")

    print("\n💡 Enjoy the Chirp 3: HD voices:")
    print("   - Naturalness and expressiveness")
    print("   - Voice quality and clarity")
    print("   - Conversation flow and timing")
    print("   - Overall realism")


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
