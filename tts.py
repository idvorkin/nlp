#!python3


import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Iterator, Optional

import typer
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from icecream import ic
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from pydub import AudioSegment
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
    # unwrapp the iterator
    audio = b"".join(audio)

    print(f"Took {round(time.time() -start,3)} seconds")
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
    voice: Annotated[
        str, typer.Option(help=f"Model any of: {list_of_voices}")
    ] = "igor",
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
def google_multi():
    # https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers#example_of_how_to_use_multi-speaker_markup
    from google.cloud import texttospeech_v1beta1

    # Define the conversation as a list of tuples (speaker, text)
    conversation = [
        ("R", "I've heard that Freddy feedback is amazing!"),
        ("S", "Oh? What's so good about it?"),
        ("R", "Well.."),
        ("S", "Well what?"),
        ("R", "Well, you should find it out by yourself!"),
        ("S", "Alright alright, let's try it out!"),
    ]

    # Instantiates a client
    client = texttospeech_v1beta1.TextToSpeechClient()
    multi_speaker_markup = texttospeech_v1beta1.MultiSpeakerMarkup()

    # Create turns from conversation data
    for speaker, text in conversation:
        turn = texttospeech_v1beta1.MultiSpeakerMarkup.Turn()
        turn.text = text
        turn.speaker = speaker
        multi_speaker_markup.turns.append(turn)

    # Set the text input to be synthesized
    synthesis_input = texttospeech_v1beta1.SynthesisInput(
        multi_speaker_markup=multi_speaker_markup
    )

    # Build the voice request, select the language code ('en-US') and the ssml
    # voice gender ('neutral')
    voice = texttospeech_v1beta1.VoiceSelectionParams(
        language_code="en-US", name="en-US-Studio-MultiSpeaker"
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech_v1beta1.AudioConfig(
        audio_encoding=texttospeech_v1beta1.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')


@app.command()
def merge_audio(directory: Path):
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


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
