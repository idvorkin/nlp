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


class Chapter(BaseModel):
    start_turn: int
    title: str
    description: str


class ConversationChapters(BaseModel):
    chapters: List[Chapter]


class ConversationState(BaseModel):
    """Persistent state for conversation generation"""

    voice_mapping: dict[str, str]
    chapters: Optional[List[Chapter]] = None
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


def analyze_conversation_chapters(
    turns: List[ConversationTurn], cache_file: Path = None, conversation_id: str = None
) -> List[Chapter]:
    """Use LLM to analyze conversation and identify chapter markers with caching"""

    # Check if we have cached results
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Validate cache against conversation structure if ID provided
            if (
                conversation_id
                and cached_data.get("conversation_id") != conversation_id
            ):
                console.print(
                    "⚠️ Cache outdated (conversation structure changed), regenerating chapters..."
                )
            else:
                chapters_model = ConversationChapters.model_validate(
                    cached_data.get("chapters", cached_data)
                )
                console.print(f"📖 Using cached chapter analysis from {cache_file}")
                return chapters_model.chapters
        except Exception as e:
            console.print(f"⚠️ Could not load cached chapters: {e}")

    # Prepare conversation text for analysis
    conversation_text = ""
    for i, turn in enumerate(turns):
        conversation_text += f"Turn {i + 1} - {turn.speaker}: {turn.text}\n"

    # Create prompt for LLM analysis
    prompt = f"""Analyze this conversation and identify logical chapter breaks with meaningful titles.

A chapter should represent a distinct topic, theme, or segment of the conversation.
Aim for 3-7 chapters for most conversations. Each chapter should have:
- A clear starting point (turn number)
- A concise, descriptive title (2-6 words)
- A brief description of what's discussed

Conversation:
{conversation_text}

Respond with JSON in this exact format:
{{
  "chapters": [
    {{
      "start_turn": 1,
      "title": "Introduction and Setup",
      "description": "Opening remarks and conversation setup"
    }},
    {{
      "start_turn": 5,
      "title": "Main Topic Discussion",
      "description": "Deep dive into the primary subject"
    }}
  ]
}}

Make sure start_turn numbers are valid (1 to {len(turns)}) and in ascending order."""

    try:
        # Use OpenAI to analyze the conversation
        import openai

        client = openai.OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing conversations and identifying logical chapter breaks. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        # Parse the JSON response
        chapters_data = json.loads(response.choices[0].message.content)
        chapters_model = ConversationChapters.model_validate(chapters_data)

        # Validate and adjust chapter start positions
        validated_chapters = []
        for chapter in chapters_model.chapters:
            # Ensure start_turn is within valid range
            start_turn = max(1, min(chapter.start_turn, len(turns)))
            validated_chapters.append(
                Chapter(
                    start_turn=start_turn,
                    title=chapter.title,
                    description=chapter.description,
                )
            )

        # Sort by start_turn and ensure first chapter starts at turn 1
        validated_chapters.sort(key=lambda x: x.start_turn)
        if not validated_chapters or validated_chapters[0].start_turn > 1:
            validated_chapters.insert(
                0,
                Chapter(
                    start_turn=1,
                    title="Introduction",
                    description="Conversation opening",
                ),
            )

        # Cache the results with conversation ID
        if cache_file:
            try:
                cache_data = {
                    "conversation_id": conversation_id,
                    "chapters": [
                        chapter.model_dump() for chapter in validated_chapters
                    ],
                }
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f, indent=2)
                console.print(f"💾 Cached chapter analysis to {cache_file}")
            except Exception as e:
                console.print(f"⚠️ Could not cache chapters: {e}")

        return validated_chapters

    except Exception as e:
        console.print(f"⚠️ Chapter analysis failed: {e}")
        console.print("📝 Using default single chapter")
        # Fallback to single chapter
        return [
            Chapter(
                start_turn=1,
                title="Full Conversation",
                description="Complete conversation without chapters",
            )
        ]


def get_cache_file_path(convo_file: Path) -> Path:
    """Get the standard cache file path for a conversation"""
    output_dir = Path.home() / "tmp/tts" / f"conversation_{convo_file.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "chapters_cache.json"


def get_audio_duration(audio_file: Path) -> float:
    """Get the duration of an audio file in seconds using ffprobe"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(audio_file),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            console.print(
                f"⚠️ Could not get duration for {audio_file}, using 4s estimate"
            )
            return 4.0  # fallback estimate
    except Exception as e:
        console.print(
            f"⚠️ Error getting duration for {audio_file}: {e}, using 4s estimate"
        )
        return 4.0  # fallback estimate


def create_ffmpeg_chapters_file(
    chapters: List[Chapter], audio_files: List[Path], output_dir: Path
) -> Path:
    """Create an ffmpeg chapters file for the conversation using actual audio durations"""

    chapters_file = output_dir / "chapters.txt"

    # Calculate actual durations for each audio file
    console.print("🕐 Calculating actual audio durations for precise chapter timing...")

    # Create a mapping from turn number to audio file
    turn_to_file = {}
    for audio_file in audio_files:
        # Extract turn number from filename like "turn_003_SPEAKER_00.mp3"
        match = re.search(r"turn_(\d+)_", audio_file.name)
        if match:
            turn_num = int(match.group(1)) + 1  # Convert 0-based to 1-based
            turn_to_file[turn_num] = audio_file

    # Calculate cumulative durations
    cumulative_duration = 0.0
    turn_start_times = {1: 0.0}  # Turn 1 starts at 0

    for turn_num in sorted(turn_to_file.keys()):
        audio_file = turn_to_file[turn_num]
        duration = get_audio_duration(audio_file)
        cumulative_duration += duration
        # Next turn starts where this one ends
        if turn_num + 1 not in turn_start_times:
            turn_start_times[turn_num + 1] = cumulative_duration
        console.print(f"   🎵 Turn {turn_num}: {audio_file.name} ({duration:.1f}s)")

    # Total conversation duration
    total_duration_ms = int(cumulative_duration * 1000)

    with open(chapters_file, "w") as f:
        f.write(";FFMETADATA1\n")

        for i, chapter in enumerate(chapters):
            # Get actual start time for this chapter
            start_seconds = turn_start_times.get(chapter.start_turn, 0.0)
            start_ms = int(start_seconds * 1000)

            # Calculate end time (start of next chapter or end of conversation)
            if i + 1 < len(chapters):
                next_start_turn = chapters[i + 1].start_turn
                end_seconds = turn_start_times.get(next_start_turn, cumulative_duration)
                end_ms = int(end_seconds * 1000)
            else:
                end_ms = total_duration_ms

            f.write("\n[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={chapter.title}\n")

            # Print timing info for debugging
            start_time_str = f"{start_seconds // 60:.0f}:{start_seconds % 60:04.1f}"
            end_time_str = f"{(end_ms / 1000) // 60:.0f}:{(end_ms / 1000) % 60:04.1f}"
            console.print(
                f"   📖 Chapter {i + 1}: {start_time_str} - {end_time_str} ({chapter.title})"
            )

    try:
        file_size = chapters_file.stat().st_size
        console.print(f"📝 Created chapters file: {chapters_file}")
        console.print(f"   File size: {file_size} bytes")
        console.print(f"   Total duration: {cumulative_duration:.1f} seconds")

        # Verify file content
        if file_size == 0:
            console.print("⚠️ WARNING: Chapters file is empty!")
        else:
            console.print("✅ Chapters file created successfully")

    except Exception as e:
        console.print(f"⚠️ Error checking chapters file: {e}")

    return chapters_file


def merge_audio_files_with_chapters(
    audio_files: List[Path], output_path: Path, chapters: List[Chapter] = None
) -> Path:
    """Merge multiple audio files into one using ffmpeg with optional chapters"""

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
    chapters_file = None

    try:
        # Create the file list
        with open(file_list_path, "w") as f:
            for audio_file in sorted(audio_files):  # Sort to ensure correct order
                if Path(audio_file).exists():
                    f.write(f"file '{audio_file.absolute()}'\n")

        # Add debug output to see what's happening with chapters
        console.print(
            f"🔍 Debug: chapters={chapters}, len={len(chapters) if chapters else 'None'}"
        )

        # Create chapters file if chapters are provided
        if chapters and len(chapters) >= 1:  # Changed from > 1 to >= 1
            try:
                chapters_file = create_ffmpeg_chapters_file(
                    chapters, audio_files, output_path.parent
                )
                if chapters_file and chapters_file.exists():
                    console.print(f"📖 Created {len(chapters)} chapters:")
                    for chapter in chapters:
                        console.print(
                            f"   • Turn {chapter.start_turn}: {chapter.title}"
                        )
                else:
                    console.print("⚠️ Failed to create chapters file")
                    chapters_file = None
            except Exception as e:
                console.print(f"⚠️ Error creating chapters: {e}")
                chapters_file = None

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
        ]

        # Add chapters if available
        if chapters_file and chapters_file.exists():
            try:
                # Give the file a moment to be fully written and check size
                import time

                time.sleep(0.1)  # Brief pause to ensure file is fully written

                file_size = chapters_file.stat().st_size
                if file_size > 0:
                    cmd.extend(["-i", str(chapters_file), "-map_metadata", "1"])
                    console.print(
                        f"📖 Adding chapters from {chapters_file} ({file_size} bytes)"
                    )
                else:
                    console.print(
                        f"⚠️ Chapters file exists but is empty: {chapters_file}"
                    )
            except Exception as e:
                console.print(f"⚠️ Error reading chapters file: {e}")
        else:
            console.print("⚠️ No valid chapters file, proceeding without chapters")

        cmd.extend(
            [
                "-c",
                "copy",
                "-y",  # Overwrite output file
                str(output_path),
            ]
        )

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
        # Only delete chapters file if merge failed
        if chapters_file and chapters_file.exists() and result.returncode != 0:
            chapters_file.unlink()


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
    chapters: bool = typer.Option(True, help="Add chapter markers to merged audio"),
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
    chapters_cache_file = get_cache_file_path(convo_file)

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
        conversation_chapters = existing_state.chapters
    else:
        console.print("🆕 Starting fresh generation...")
        # Get unique speakers and create voice mapping
        speakers = {turn.speaker for turn in turns}
        voice_mapping = get_voice_mapping(speakers)

        # Analyze chapters if requested
        conversation_chapters = None
        if (
            chapters and len(turns) > 3
        ):  # Only analyze chapters for longer conversations
            console.print("\n🧠 Analyzing conversation for chapter markers...")
            conversation_chapters = analyze_conversation_chapters(
                turns, chapters_cache_file, conversation_hash
            )

        # Create new state
        existing_state = ConversationState(
            voice_mapping=voice_mapping,
            chapters=conversation_chapters,
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
        merged_file = merge_audio_files_with_chapters(
            audio_files, output_dir / "full_conversation.mp3", conversation_chapters
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
        if conversation_chapters and len(conversation_chapters) > 1:
            console.print(
                f"   • Chapters: {len(conversation_chapters)} chapter markers added"
            )

    # Clean up state file on successful completion
    if len(existing_state.completed_turns) == len(turns):
        console.print("🧹 Cleaning up state file (generation complete)")
        if state_file.exists():
            state_file.unlink()


def merge_audio_files(audio_files: List[Path], output_path: Path) -> Path:
    """Legacy function - redirects to new chapter-aware function"""
    return merge_audio_files_with_chapters(audio_files, output_path, None)


@app.command()
def chapters_only(
    convo_file: Path = typer.Argument(
        Path("samples/convo.1.txt"), help="Path to conversation file"
    ),
):
    """Analyze conversation and display chapter markers without generating audio"""

    if not convo_file.exists():
        console.print(f"❌ Conversation file not found: {convo_file}")
        raise typer.Exit(1)

    # Parse the conversation
    console.print(f"📖 Reading conversation from: {convo_file}")
    turns = parse_conversation_file(convo_file)

    if not turns:
        console.print("❌ No conversation turns found in file")
        raise typer.Exit(1)

    console.print(f"💬 Found {len(turns)} conversation turns")

    # Analyze chapters with caching
    chapters_cache_file = get_cache_file_path(convo_file)

    # Get conversation identifier for cache validation
    conversation_hash = get_conversation_identifier(convo_file, turns)
    console.print(f"🔍 Conversation hash: {conversation_hash[:8]}...")

    console.print("\n🧠 Analyzing conversation for chapter markers...")
    chapters = analyze_conversation_chapters(
        turns, chapters_cache_file, conversation_hash
    )

    console.print("\n📖 Chapter Analysis Results:")
    console.print(f"   • Total chapters: {len(chapters)}")
    console.print(f"   • Conversation length: {len(turns)} turns")

    # Display chapters with context
    console.print("\n📚 Chapter Breakdown:")

    for i, chapter in enumerate(chapters):
        console.print("\n" + "=" * 60)
        console.print(f"📖 Chapter {i + 1}: {chapter.title}")
        console.print(f"   Description: {chapter.description}")
        console.print(f"   Starts at turn: {chapter.start_turn}")

        # Show a few turns from this chapter
        chapter_end = (
            chapters[i + 1].start_turn - 1 if i + 1 < len(chapters) else len(turns)
        )
        turns_in_chapter = chapter_end - chapter.start_turn + 1
        console.print(f"   Duration: {turns_in_chapter} turns")

        console.print("\n   📝 Sample content:")
        # Show first 3 turns of each chapter
        for j in range(min(3, turns_in_chapter)):
            turn_idx = chapter.start_turn - 1 + j  # Convert to 0-based index
            if turn_idx < len(turns):
                turn = turns[turn_idx]
                turn_number = turn_idx + 1
                console.print(
                    f"      Turn {turn_number} - {turn.speaker}: {turn.text[:80]}{'...' if len(turn.text) > 80 else ''}"
                )

        if turns_in_chapter > 3:
            console.print(
                f"      ... and {turns_in_chapter - 3} more turns in this chapter"
            )

    console.print("\n" + "=" * 60)
    console.print("📊 Summary:")
    console.print(f"   • File: {convo_file}")
    console.print(f"   • Total turns: {len(turns)}")
    console.print(f"   • Chapters identified: {len(chapters)}")
    console.print(
        f"   • Average chapter length: {len(turns) / len(chapters):.1f} turns"
    )

    console.print("\n💡 To generate audio with these chapters:")
    console.print(f"   uv run tts-dialog.py recreate {convo_file}")


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
