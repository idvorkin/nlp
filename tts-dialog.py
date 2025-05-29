#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "icecream",
#     "loguru",
#     "pydantic",
#     "rich",
#     "google-cloud-texttospeech",
# ]
# ///

import asyncio
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import typer
from google.cloud import texttospeech as tts
from google.cloud.texttospeech import (
    AudioEncoding,
    VoiceSelectionParams,
    SynthesisInput,
    AudioConfig,
)
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

console = Console()
app = typer.Typer(no_args_is_help=True)


class ConversationTurn(BaseModel):
    speaker: str
    text: str


class ConversationConfig(BaseModel):
    turns: List[ConversationTurn]
    voice_mapping: dict[str, str]


def parse_conversation_file(file_path: Path) -> List[ConversationTurn]:
    """Parse a conversation file with [SPEAKER_XX]: format"""
    turns = []

    with open(file_path, "r") as f:
        content = f.read()

    # Pattern to match [SPEAKER_XX]: followed by text
    pattern = r"\[SPEAKER_(\d+)\]:\s*(.*?)(?=\[SPEAKER_|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    for speaker_id, text in matches:
        # Clean up the text - remove extra whitespace and newlines
        cleaned_text = " ".join(text.strip().split())
        if cleaned_text:  # Only add non-empty turns
            turns.append(
                ConversationTurn(speaker=f"SPEAKER_{speaker_id}", text=cleaned_text)
            )

    return turns


def get_voice_mapping(speakers: set[str]) -> dict[str, str]:
    """Map speakers to different Chirp 3: HD voices"""

    # Available Chirp 3: HD voices for en-US
    # Categorized by typical gender associations of the voice characteristics
    male_voices = [
        "en-US-Chirp3-HD-Charon",
        "en-US-Chirp3-HD-Fenrir",
        "en-US-Chirp3-HD-Gacrux",
        "en-US-Chirp3-HD-Iapetus",
        "en-US-Chirp3-HD-Orus",
        "en-US-Chirp3-HD-Puck",
        "en-US-Chirp3-HD-Rasalgethi",
        "en-US-Chirp3-HD-Schedar",
        "en-US-Chirp3-HD-Alnilam",
        "en-US-Chirp3-HD-Achernar",
    ]

    female_voices = [
        "en-US-Chirp3-HD-Leda",
        "en-US-Chirp3-HD-Aoede",
        "en-US-Chirp3-HD-Autonoe",
        "en-US-Chirp3-HD-Callirrhoe",
        "en-US-Chirp3-HD-Despina",
        "en-US-Chirp3-HD-Erinome",
        "en-US-Chirp3-HD-Laomedeia",
        "en-US-Chirp3-HD-Pulcherrima",
        "en-US-Chirp3-HD-Vindemiatrix",
        "en-US-Chirp3-HD-Algieba",
    ]

    # All voices combined for fallback
    all_voices = [
        "en-US-Chirp3-HD-Achernar",
        "en-US-Chirp3-HD-Achird",
        "en-US-Chirp3-HD-Algenib",
        "en-US-Chirp3-HD-Algieba",
        "en-US-Chirp3-HD-Alnilam",
        "en-US-Chirp3-HD-Aoede",
        "en-US-Chirp3-HD-Autonoe",
        "en-US-Chirp3-HD-Callirrhoe",
        "en-US-Chirp3-HD-Charon",
        "en-US-Chirp3-HD-Despina",
        "en-US-Chirp3-HD-Enceladus",
        "en-US-Chirp3-HD-Erinome",
        "en-US-Chirp3-HD-Fenrir",
        "en-US-Chirp3-HD-Gacrux",
        "en-US-Chirp3-HD-Iapetus",
        "en-US-Chirp3-HD-Kore",
        "en-US-Chirp3-HD-Laomedeia",
        "en-US-Chirp3-HD-Leda",
        "en-US-Chirp3-HD-Orus",
        "en-US-Chirp3-HD-Puck",
        "en-US-Chirp3-HD-Pulcherrima",
        "en-US-Chirp3-HD-Rasalgethi",
        "en-US-Chirp3-HD-Sadachbia",
        "en-US-Chirp3-HD-Sadaltager",
        "en-US-Chirp3-HD-Schedar",
        "en-US-Chirp3-HD-Sulafat",
        "en-US-Chirp3-HD-Umbriel",
        "en-US-Chirp3-HD-Vindemiatrix",
        "en-US-Chirp3-HD-Zephyr",
        "en-US-Chirp3-HD-Zubenelgenubi",
    ]

    voice_mapping = {}
    sorted_speakers = sorted(speakers)

    # Special case: exactly 2 speakers - assign male and female voices
    if len(speakers) == 2:
        voice_mapping[sorted_speakers[0]] = male_voices[
            0
        ]  # First speaker gets male voice
        voice_mapping[sorted_speakers[1]] = female_voices[
            0
        ]  # Second speaker gets female voice
        console.print(
            f"   🎭 Gender-diverse casting: {sorted_speakers[0]} (male voice), {sorted_speakers[1]} (female voice)"
        )
    else:
        # For other cases, use the full list with variety
        for i, speaker in enumerate(sorted_speakers):
            voice_mapping[speaker] = all_voices[i % len(all_voices)]

    return voice_mapping


async def generate_single_turn(
    turn: ConversationTurn, voice_name: str, output_path: Path, speed: float = 1.0
):
    """Generate a single conversation turn asynchronously"""
    client = tts.TextToSpeechAsyncClient()

    try:
        synthesis_input = SynthesisInput(text=turn.text)

        audio_config = AudioConfig(
            audio_encoding=AudioEncoding.MP3,
            speaking_rate=speed,
        )

        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=VoiceSelectionParams(language_code="en-US", name=voice_name),
            audio_config=audio_config,
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        console.print(f"   ✅ Generated: {output_path}")
        return output_path

    except Exception as e:
        console.print(f"   ❌ Failed to generate {output_path}: {e}")
        return None


@app.command()
def recreate(
    convo_file: Path = typer.Argument(
        Path("samples/convo.1.txt"), help="Path to conversation file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, help="Output directory for audio files"
    ),
    speed: float = typer.Option(1.0, help="Speaking speed (0.5-2.0)"),
    speak: bool = typer.Option(True, help="Play the conversation after generation"),
    merge: bool = typer.Option(
        True, help="Merge all audio files into one conversation"
    ),
):
    """Recreate a conversation using Google Chirp 3: HD voices"""

    if not convo_file.exists():
        console.print(f"❌ Conversation file not found: {convo_file}")
        raise typer.Exit(1)

    # Parse the conversation
    console.print(f"📖 Reading conversation from: {convo_file}")
    turns = parse_conversation_file(convo_file)

    if not turns:
        console.print("❌ No conversation turns found in file")
        raise typer.Exit(1)

    # Get unique speakers and create voice mapping
    speakers = {turn.speaker for turn in turns}
    voice_mapping = get_voice_mapping(speakers)

    console.print(f"\n🎭 Found {len(speakers)} speakers:")
    for speaker, voice in voice_mapping.items():
        console.print(f"   {speaker} → {voice}")

    # Set up output directory
    if output_dir is None:
        output_dir = Path.home() / "tmp/tts" / f"conversation_{convo_file.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n📁 Output directory: {output_dir}")
    console.print(f"🔊 Speed: {speed}x")
    console.print("\n💬 Conversation preview:")
    for i, turn in enumerate(turns[:3]):  # Show first 3 turns
        console.print(
            f"   {turn.speaker}: {turn.text[:60]}{'...' if len(turn.text) > 60 else ''}"
        )
    if len(turns) > 3:
        console.print(f"   ... and {len(turns) - 3} more turns")

    # Generate all audio files
    async def generate_all_turns():
        semaphore = asyncio.Semaphore(50)  # Limit concurrent requests

        async def generate_with_semaphore(turn, i):
            async with semaphore:
                output_path = output_dir / f"turn_{i:03d}_{turn.speaker}.mp3"
                voice_name = voice_mapping[turn.speaker]
                return await generate_single_turn(turn, voice_name, output_path, speed)

        tasks = [generate_with_semaphore(turn, i) for i, turn in enumerate(turns)]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    console.print(f"\n🚀 Generating {len(turns)} audio segments...")
    start_time = time.time()

    audio_files = asyncio.run(generate_all_turns())
    generation_time = time.time() - start_time

    console.print(
        f"✅ Generated {len(audio_files)} files in {generation_time:.2f} seconds"
    )

    # Merge audio files if requested
    merged_file = None
    if merge and audio_files:
        console.print("\n🔗 Merging audio files...")
        merged_file = merge_audio_files(
            audio_files, output_dir / "full_conversation.mp3"
        )

    # Play the conversation
    if speak:
        if merged_file and merged_file.exists():
            console.print("\n🔊 Playing merged conversation...")
            subprocess.run(["afplay", merged_file])
        else:
            console.print("\n🔊 Playing individual segments...")
            for audio_file in audio_files:
                if Path(audio_file).exists():
                    subprocess.run(["afplay", audio_file])
                    time.sleep(0.3)  # Small pause between segments

    console.print("\n📊 Summary:")
    console.print(f"   • Generated: {len(audio_files)} audio files")
    console.print(f"   • Time taken: {generation_time:.2f} seconds")
    console.print(f"   • Output: {output_dir}")
    if merged_file:
        console.print(f"   • Merged: {merged_file}")


def merge_audio_files(audio_files: List[Path], output_path: Path) -> Path:
    """Merge multiple audio files into one using ffmpeg"""

    # Create a file list for ffmpeg
    file_list_path = output_path.parent / "filelist.txt"

    try:
        # Create the file list
        with open(file_list_path, "w") as f:
            for audio_file in sorted(audio_files):  # Sort to ensure correct order
                if Path(audio_file).exists():
                    f.write(f"file '{audio_file.absolute()}'\n")

        # Use ffmpeg to concatenate the files
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            console.print(f"❌ FFmpeg error: {result.stderr}")
            return None

        return output_path

    except Exception as e:
        console.print(f"❌ Error merging audio files: {e}")
        return None

    finally:
        # Clean up the temporary file list
        if file_list_path.exists():
            file_list_path.unlink()


@app.command()
def list_voices():
    """List available Chirp 3: HD voices"""
    voices = [
        "en-US-Chirp3-HD-Achernar",
        "en-US-Chirp3-HD-Achird",
        "en-US-Chirp3-HD-Algenib",
        "en-US-Chirp3-HD-Algieba",
        "en-US-Chirp3-HD-Alnilam",
        "en-US-Chirp3-HD-Aoede",
        "en-US-Chirp3-HD-Autonoe",
        "en-US-Chirp3-HD-Callirrhoe",
        "en-US-Chirp3-HD-Charon",
        "en-US-Chirp3-HD-Despina",
        "en-US-Chirp3-HD-Enceladus",
        "en-US-Chirp3-HD-Erinome",
        "en-US-Chirp3-HD-Fenrir",
        "en-US-Chirp3-HD-Gacrux",
        "en-US-Chirp3-HD-Iapetus",
        "en-US-Chirp3-HD-Kore",
        "en-US-Chirp3-HD-Laomedeia",
        "en-US-Chirp3-HD-Leda",
        "en-US-Chirp3-HD-Orus",
        "en-US-Chirp3-HD-Puck",
        "en-US-Chirp3-HD-Pulcherrima",
        "en-US-Chirp3-HD-Rasalgethi",
        "en-US-Chirp3-HD-Sadachbia",
        "en-US-Chirp3-HD-Sadaltager",
        "en-US-Chirp3-HD-Schedar",
        "en-US-Chirp3-HD-Sulafat",
        "en-US-Chirp3-HD-Umbriel",
        "en-US-Chirp3-HD-Vindemiatrix",
        "en-US-Chirp3-HD-Zephyr",
        "en-US-Chirp3-HD-Zubenelgenubi",
    ]

    console.print("🎙️ Available Chirp 3: HD voices:")
    for voice in voices:
        console.print(f"   • {voice}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
