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
#     "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git",
#     "simple-term-menu",
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


MLX_MODELS = [
    {
        "name": "Qwen3-TTS-0.6B",
        "path": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "size": "0.6B",
        "released": "2026-01",
        "features": "Voice cloning, 8-bit quantized",
    },
    {
        "name": "Qwen3-TTS-1.7B",
        "path": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "size": "1.7B",
        "released": "2026-01",
        "features": "Voice design, emotion control",
        "instruct": "A warm, friendly male voice with natural conversational tone",
    },
    {
        "name": "Marvis",
        "path": "Marvis-AI/marvis-tts-250m-v0.1",
        "size": "250M",
        "released": "2025-08",
        "features": "Real-time streaming",
    },
    {
        "name": "Chatterbox",
        "path": "mlx-community/Chatterbox-bf16",
        "size": "1B",
        "released": "2025-05",
        "features": "Expressive, multilingual",
    },
    {
        "name": "Dia",
        "path": "mlx-community/Dia-1.6B-bf16",
        "size": "1.6B",
        "released": "2025-04",
        "features": "Dialogue-focused",
    },
    {
        "name": "CSM",
        "path": "mlx-community/csm-1b",
        "size": "1B",
        "released": "2025-03",
        "features": "Conversational, voice cloning",
    },
    {
        "name": "SparkTTS",
        "path": "mlx-community/SparkTTS-0.5B-bf16",
        "size": "0.5B",
        "released": "2025-03",
        "features": "Bilingual EN/Mandarin",
    },
    {
        "name": "Kokoro-82M",
        "path": "mlx-community/Kokoro-82M-bf16",
        "size": "82M",
        "released": "2025-01",
        "features": "Fast, multilingual, multiple voices",
    },
    {
        "name": "OuteTTS",
        "path": "mlx-community/OuteTTS-0.2-500M",
        "size": "500M",
        "released": "2024-11",
        "features": "Efficient, lightweight",
    },
    {
        "name": "F5-TTS",
        "path": "lucasnewman/f5-tts-mlx",
        "size": "300M",
        "released": "2024-10",
        "features": "Zero-shot, flow-matching",
    },
]


@app.command()
def mlx_models():
    """List available MLX TTS models for local inference on Apple Silicon"""
    from rich.table import Table

    table = Table(title="MLX TTS Models (Apple Silicon)")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Released", style="yellow")
    table.add_column("Features", style="dim")

    for m in MLX_MODELS:
        table.add_row(m["name"], m["size"], m["released"], m["features"])

    console.print(table)
    console.print("\n[dim]Use with:[/dim] tts.py mlx --model <path> \"text\"")
    console.print("[dim]Example:[/dim] tts.py mlx --model mlx-community/Kokoro-82M-bf16 \"Hello\"")


def _list_audio_devices() -> list[tuple[int, str]]:
    """List available audio input devices on macOS"""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    devices = []
    in_audio = False
    for line in result.stderr.splitlines():
        if "audio devices" in line.lower():
            in_audio = True
            continue
        if in_audio and "[" in line and "]" in line:
            # Parse lines like: [AVFoundation indev @ 0x...] [0] MacBook Air Microphone
            match = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if match:
                devices.append((int(match.group(1)), match.group(2)))
    return devices


def _get_mlx_output_file(base_path: Path) -> Path:
    """Get actual output file path - mlx-audio adds _000 suffix"""
    actual = base_path.with_suffix("").with_name(f"{base_path.stem}_000.wav")
    return actual if actual.exists() else base_path


@app.command()
def record(
    outfile: Annotated[Optional[Path], typer.Argument(help="Output WAV file")] = None,
    duration: int = 10,
    mic: Annotated[Optional[int], typer.Option(help="Microphone device index")] = None,
):
    """Record voice from microphone for voice cloning (press Ctrl+C to stop early)"""
    from simple_term_menu import TerminalMenu

    if outfile is None:
        outfile = Path.home() / "tmp/tts" / "my_voice.wav"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Pick microphone if not specified
    if mic is None:
        devices = _list_audio_devices()
        if not devices:
            console.print("[red]No audio input devices found[/red]")
            raise typer.Exit(1)

        options = [f"[{idx}] {name}" for idx, name in devices]
        menu = TerminalMenu(options, title="Select microphone:")
        choice = menu.show()
        if choice is None:
            raise typer.Exit(0)
        mic = devices[choice][0]
        console.print(f"[blue]Using:[/blue] {devices[choice][1]}\n")

    console.print(f"[blue]Recording for up to {duration} seconds...[/blue]")
    console.print("[dim]Press Ctrl+C to stop early[/dim]\n")
    console.print("[bold red]🎤 Recording...[/bold red]")

    try:
        # Use ffmpeg to record from selected audio input
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite
                "-f", "avfoundation",
                "-i", f":{mic}",  # Selected audio input on macOS
                "-t", str(duration),
                "-ar", "24000",  # 24kHz sample rate (good for TTS)
                "-ac", "1",  # Mono
                str(outfile),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "Exiting normally" not in result.stderr:
            console.print(f"[red]Error:[/red] {result.stderr[:200]}")
            raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped early[/yellow]")

    if outfile.exists():
        console.print(f"\n[green]Saved:[/green] {outfile}")
        console.print(f"[dim]Use for voice cloning with:[/dim] tts.py mlx-clone")
    else:
        console.print("[red]Recording failed[/red]")


@app.command()
def mlx_clone(
    voice: Annotated[Path, typer.Option(help="WAV file of voice to clone")] = Path.home()
    / "tmp/tts/my_voice.wav",
    text: Annotated[Optional[str], typer.Argument(help="Text to speak")] = None,
):
    """Clone your voice using MLX TTS - interactive mode"""
    if not voice.exists():
        console.print(f"[red]Voice file not found:[/red] {voice}")
        console.print("[dim]Record your voice first with:[/dim] tts.py record")
        raise typer.Exit(1)

    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio as mlx_generate

    console.print(f"[blue]Voice:[/blue] {voice}")

    model_path = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
    console.print(f"[dim]Loading model...[/dim]")
    tts_model = load_model(model_path)
    console.print(f"[green]Model loaded[/green]\n")

    while True:
        if text is None:
            text = typer.prompt("What should your clone say?")
        if not text:
            break

        console.print(f"[dim]Generating...[/dim]")
        temp_path = Path.home() / "tmp/tts" / f"clone_{random.random()}.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        gen_start = time.time()
        mlx_generate(
            model=tts_model,
            text=text,
            file_prefix=str(temp_path.with_suffix("")),
            audio_prompt=str(voice),
        )
        gen_time = time.time() - gen_start

        temp_path = _get_mlx_output_file(temp_path)

        console.print(f"[green]Generated in {gen_time:.2f}s[/green]")
        subprocess.run(["afplay", str(temp_path)])

        # Prompt for next
        text = None
        another = typer.prompt("\nAnother? (Enter text or 'q' to quit)", default="")
        if another.lower() == "q" or another == "":
            break
        text = another


@app.command()
def mlx_try(
    text: str = "Hello! This is a test of the text to speech system. How does it sound?",
):
    """Interactive TUI to try different MLX TTS models (arrow keys to navigate)"""
    import warnings
    from simple_term_menu import TerminalMenu

    warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")

    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio as mlx_generate

    while True:
        # Build menu options
        options = [f"{m['name']} ({m['size']}) - {m['features']}" for m in MLX_MODELS]
        options.append("[Exit]")

        console.print(f"\n[dim]Text:[/dim] {text[:70]}{'...' if len(text) > 70 else ''}\n")

        menu = TerminalMenu(
            options,
            title="Select MLX TTS Model (↑↓ to navigate, Enter to select):",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("bg_cyan", "fg_black"),
        )
        idx = menu.show()

        if idx is None or idx == len(MLX_MODELS):
            break

        model_info = MLX_MODELS[idx]
        model_path = model_info["path"]
        console.print(f"\n[blue]Loading {model_info['name']}...[/blue]")

        start = time.time()
        tts_model = load_model(model_path)
        load_time = time.time() - start
        console.print(f"[dim]Model loaded in {load_time:.2f}s[/dim]")

        gen_start = time.time()
        temp_path = Path.home() / "tmp/tts" / f"mlx_try_{model_info['name']}.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        gen_kwargs = {"model": tts_model, "text": text, "file_prefix": str(temp_path.with_suffix(""))}

        # VoiceDesign models need a voice style description
        if "instruct" in model_info:
            voice_styles = [
                "A warm, friendly male voice with natural conversational tone",
                "A cheerful young female voice with high pitch",
                "A calm, soothing female voice for meditation",
                "A deep, authoritative male voice for narration",
                "An energetic, enthusiastic voice for announcements",
                "A gentle, caring voice like a teacher",
                "A professional news anchor voice",
                "A gruff, deep Italian-American male voice from New Jersey with a tough guy attitude",
                "A wise elderly British man with a distinguished, measured tone",
                "A fast-talking New York City radio DJ voice",
                "Custom...",
            ]
            console.print()
            style_menu = TerminalMenu(voice_styles, title="Select voice style:")
            style_idx = style_menu.show()
            if style_idx is None:
                continue
            if style_idx == len(voice_styles) - 1:
                instruct = typer.prompt("Describe the voice")
            else:
                instruct = voice_styles[style_idx]
            gen_kwargs["instruct"] = instruct
            console.print(f"[dim]Voice style:[/dim] {instruct}")

        mlx_generate(**gen_kwargs)
        gen_time = time.time() - gen_start

        temp_path = _get_mlx_output_file(temp_path)

        console.print(f"[green]Generated in {gen_time:.2f}s[/green]")
        subprocess.run(["afplay", str(temp_path)])

        # Post-playback menu
        post_options = ["Try another model", "Change text", "Exit"]
        post_menu = TerminalMenu(post_options, title="\nWhat next?")
        post_choice = post_menu.show()

        if post_choice == 2 or post_choice is None:
            break
        elif post_choice == 1:
            new_text = typer.prompt("Enter new text")
            if new_text:
                text = new_text


@app.command()
def mlx(
    text: Annotated[Optional[str], typer.Argument(help="Text to speak")] = None,
    model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    voice: Annotated[Optional[Path], typer.Option(help="WAV file for voice cloning")] = None,
    speak: bool = True,
    outfile: Optional[Path] = None,
):
    """Local TTS using MLX on Apple Silicon (no API keys needed)

    Run 'tts.py mlx-models' to see all available models.
    Use --voice with a WAV file to clone a voice.
    """
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio as mlx_generate

    if text is None:
        text = "\n".join(sys.stdin.readlines()).strip()
    if not text:
        console.print("[red]No text provided[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Model:[/blue] {model}")
    if voice:
        console.print(f"[blue]Voice:[/blue] {voice}")
    console.print(f"[blue]Text:[/blue] {text[:80]}{'...' if len(text) > 80 else ''}")

    start = time.time()
    tts_model = load_model(model)
    load_time = time.time() - start
    console.print(f"[dim]Model loaded in {load_time:.2f}s[/dim]")

    gen_start = time.time()
    if outfile is None:
        temp_path = Path.home() / "tmp/tts" / f"mlx_{random.random()}.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        outfile = temp_path

    gen_kwargs = {"model": tts_model, "text": text, "file_prefix": str(outfile.with_suffix(""))}
    if voice and voice.exists():
        gen_kwargs["audio_prompt"] = str(voice)

    mlx_generate(**gen_kwargs)
    gen_time = time.time() - gen_start

    outfile = _get_mlx_output_file(outfile)

    console.print(f"[green]Generated in {gen_time:.2f}s[/green] → {outfile}")

    if speak:
        subprocess.run(["afplay", str(outfile)])


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
