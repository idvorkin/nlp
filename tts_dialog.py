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
#     "openai",
# ]
# ///

import asyncio
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import json

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


class ConversationState(BaseModel):
    """Persistent state for conversation generation"""

    voice_mapping: dict[str, str]
    completed_turns: List[int] = []
    total_turns: int
    speed: float
    conversation_hash: str  # To detect if source file changed
    merged_turn_count: int  # To detect if merging changed the structure


def get_conversation_identifier(
    file_path: Path, merged_turns: List[ConversationTurn]
) -> str:
    """Get a unique identifier for the conversation including merge information"""
    import hashlib

    # Include file content and the structure after merging
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Create a structure hash that includes speaker sequence and turn count
    structure_info = f"{len(merged_turns)}:" + ":".join(
        turn.speaker for turn in merged_turns
    )

    combined_data = file_content + structure_info.encode()
    return hashlib.md5(combined_data).hexdigest()


def get_file_hash(file_path: Path) -> str:
    """Get a hash of the file content to detect changes"""
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_conversation_state(state_file: Path) -> Optional[ConversationState]:
    """Load conversation state from JSON file"""
    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            data = json.load(f)
        return ConversationState.model_validate(data)
    except Exception as e:
        console.print(f"⚠️ Could not load state file: {e}")
        return None


def save_conversation_state(state: ConversationState, state_file: Path):
    """Save conversation state to JSON file"""
    try:
        with open(state_file, "w") as f:
            json.dump(state.model_dump(), f, indent=2)
    except Exception as e:
        console.print(f"⚠️ Could not save state file: {e}")


def parse_conversation_file(file_path: Path) -> List[ConversationTurn]:
    """Parse a conversation file with [SPEAKER_XX]: format and merge consecutive turns from same speaker"""
    raw_turns = []

    with open(file_path, "r") as f:
        content = f.read()

    # Pattern to match [SPEAKER_XX]: followed by text
    pattern = r"\[SPEAKER_(\d+)\]:\s*(.*?)(?=\[SPEAKER_|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    # First pass: collect all raw turns
    for speaker_id, text in matches:
        # Clean up the text - remove extra whitespace and newlines
        cleaned_text = " ".join(text.strip().split())
        if cleaned_text:  # Only add non-empty turns
            raw_turns.append(
                ConversationTurn(speaker=f"SPEAKER_{speaker_id}", text=cleaned_text)
            )

    # Second pass: merge consecutive turns from same speaker
    merged_turns = []
    for turn in raw_turns:
        if merged_turns and merged_turns[-1].speaker == turn.speaker:
            # Merge with previous turn from same speaker
            merged_turns[-1].text += " " + turn.text
        else:
            # New speaker or first turn
            merged_turns.append(turn)

    console.print(
        f"📝 Merged {len(raw_turns)} raw turns into {len(merged_turns)} conversation turns"
    )
    if len(raw_turns) != len(merged_turns):
        console.print(
            f"   ♻️ Combined {len(raw_turns) - len(merged_turns)} consecutive same-speaker turns"
        )

    return merged_turns


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
        "en-US-Chirp3-HD-Autonoe",
        "en-US-Chirp3-HD-Leda",
        "en-US-Chirp3-HD-Aoede",
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
    turn: ConversationTurn,
    voice_name: str,
    output_path: Path,
    speed: float = 1.0,
    client=None,
):
    """Generate a single conversation turn asynchronously"""

    # Check if file already exists and is non-empty
    if output_path.exists() and output_path.stat().st_size > 0:
        console.print(f"   ✅ Already exists: {output_path}")
        return output_path

    # Use provided client or create a new one
    if client is None:
        client = tts.TextToSpeechAsyncClient()
        should_close_client = True
    else:
        should_close_client = False

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

        # Use context manager to ensure file is properly closed
        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        console.print(f"   ✅ Generated: {output_path}")
        return output_path

    except Exception as e:
        console.print(f"   ❌ Failed to generate {output_path}: {e}")
        return None
    finally:
        # Only close client if we created it locally
        if should_close_client:
            try:
                await client.transport.close()
            except Exception:
                pass  # Best effort cleanup


def merge_audio_files(audio_files: List[Path], output_path: Path) -> Path:
    """Merge multiple audio files into one using ffmpeg"""

    # Check if merged file already exists and is recent
    if output_path.exists() and output_path.stat().st_size > 0:
        # Check if any audio files are newer than the merged file
        merged_mtime = output_path.stat().st_mtime
        if all(
            audio_file.stat().st_mtime <= merged_mtime
            for audio_file in audio_files
            if audio_file.exists()
        ):
            console.print(f"✅ Merged file already up to date: {output_path}")
            return output_path

    # Create a file list for ffmpeg
    file_list_path = output_path.parent / "filelist.txt"

    try:
        # Create the file list
        with open(file_list_path, "w") as f:
            for audio_file in sorted(audio_files):  # Sort to ensure correct order
                if Path(audio_file).exists():
                    f.write(f"file '{audio_file.absolute()}'\n")

        # Build ffmpeg command
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
        # Clean up temporary files
        if file_list_path.exists():
            file_list_path.unlink()


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
    force: bool = typer.Option(False, help="Force regeneration of all files"),
):
    """Recreate a conversation using Google Chirp 3: HD voices"""

    if not convo_file.exists():
        console.print(f"❌ Conversation file not found: {convo_file}")
        raise typer.Exit(1)

    # Set up output directory
    if output_dir is None:
        output_dir = Path.home() / "tmp/tts" / f"conversation_{convo_file.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # State management
    state_file = output_dir / "conversation_state.json"

    # Parse the conversation first to get the actual structure
    console.print(f"📖 Reading conversation from: {convo_file}")
    turns = parse_conversation_file(convo_file)

    if not turns:
        console.print("❌ No conversation turns found in file")
        raise typer.Exit(1)

    # Get conversation identifier including merge information
    conversation_hash = get_conversation_identifier(convo_file, turns)
    console.print(f"🔍 Conversation hash: {conversation_hash[:8]}...")

    # Load existing state
    existing_state = load_conversation_state(state_file) if not force else None

    # Check if we can reuse existing state
    can_reuse_state = (
        existing_state is not None
        and existing_state.conversation_hash == conversation_hash
        and existing_state.total_turns == len(turns)
        and existing_state.speed == speed
        and existing_state.merged_turn_count == len(turns)
    )

    if can_reuse_state:
        console.print("🔄 Resuming from existing state...")
        voice_mapping = existing_state.voice_mapping
    else:
        console.print("🆕 Starting fresh generation...")
        # Get unique speakers and create voice mapping
        speakers = {turn.speaker for turn in turns}
        voice_mapping = get_voice_mapping(speakers)

        # Create new state
        existing_state = ConversationState(
            voice_mapping=voice_mapping,
            completed_turns=[],
            total_turns=len(turns),
            speed=speed,
            conversation_hash=conversation_hash,
            merged_turn_count=len(turns),
        )

    console.print(f"\n🎭 Found {len(set(turn.speaker for turn in turns))} speakers:")
    for speaker, voice in voice_mapping.items():
        console.print(f"   {speaker} → {voice}")

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
        # Use just one shared client to be maximally conservative
        # But allow high concurrency since these are async calls through one client
        semaphore = asyncio.Semaphore(25)  # Back to 50 for performance

        # Create a single shared client
        client = tts.TextToSpeechAsyncClient()
        try:

            async def generate_with_semaphore(turn, i):
                async with semaphore:
                    output_path = output_dir / f"turn_{i:03d}_{turn.speaker}.mp3"
                    voice_name = voice_mapping[turn.speaker]

                    result = await generate_single_turn(
                        turn, voice_name, output_path, speed, client
                    )

                    # Update state on successful generation
                    if result is not None and i not in existing_state.completed_turns:
                        existing_state.completed_turns.append(i)
                        save_conversation_state(existing_state, state_file)

                    return result

            tasks = [generate_with_semaphore(turn, i) for i, turn in enumerate(turns)]
            results = await asyncio.gather(*tasks)
            return [result for result in results if result is not None]

        finally:
            # Clean up the single client
            try:
                await client.transport.close()
            except Exception:
                pass  # Best effort cleanup

    console.print(f"\n🚀 Generating {len(turns)} audio segments...")
    if can_reuse_state and existing_state.completed_turns:
        console.print(f"   ♻️ {len(existing_state.completed_turns)} already completed")

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

    # Clean up state file on successful completion
    if len(existing_state.completed_turns) == len(turns):
        console.print("🧹 Cleaning up state file (generation complete)")
        if state_file.exists():
            state_file.unlink()


@app.command()
def clean(
    convo_file: Path = typer.Argument(
        Path("samples/convo.1.txt"), help="Path to conversation file"
    ),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory to clean"),
):
    """Clean up all generated files and state for a conversation"""

    if output_dir is None:
        output_dir = Path.home() / "tmp/tts" / f"conversation_{convo_file.stem}"

    if not output_dir.exists():
        console.print(f"📁 Directory doesn't exist: {output_dir}")
        return

    # List what we'll delete
    state_files = list(output_dir.glob("*.json"))
    audio_files = list(output_dir.glob("*.mp3"))
    temp_files = list(output_dir.glob("*.txt"))

    total_files = len(state_files) + len(audio_files) + len(temp_files)

    if total_files == 0:
        console.print(f"📁 Directory is already clean: {output_dir}")
        return

    console.print(f"🧹 Cleaning up {total_files} files from {output_dir}")
    console.print(f"   • State files: {len(state_files)}")
    console.print(f"   • Audio files: {len(audio_files)}")
    console.print(f"   • Temp files: {len(temp_files)}")

    # Clean up files
    for file_list in [state_files, audio_files, temp_files]:
        for file_path in file_list:
            try:
                file_path.unlink()
            except Exception as e:
                console.print(f"⚠️ Could not delete {file_path}: {e}")

    # Remove directory if empty
    try:
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
            console.print(f"📁 Removed empty directory: {output_dir}")
    except Exception:
        pass  # Directory not empty or other issue

    console.print("✅ Cleanup complete")


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
