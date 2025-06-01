#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "rich",
#     "deepgram-sdk",
#     "pydantic",
#     "icecream",
#     "loguru",
#     "langchain",
#     "langchain-core",
#     "langchain-openai",
#     "langchain-google-genai",
# ]
# ///

import os
import time
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from typing_extensions import Annotated
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from loguru import logger
from rich.console import Console
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

import langchain_helper

console = Console()
app = typer.Typer(
    name="diarize",
    help="Convert audio files to structured conversations using speaker diarization with Deepgram",
    add_completion=False,
    no_args_is_help=True,
)


class Chapter(BaseModel):
    """Data model for a chapter with timing information"""

    title: str
    start_seconds: float
    end_seconds: float


class ChapterResponse(BaseModel):
    """Data model for the complete chapter generation response"""

    chapters: List[Chapter]
    reasoning: Optional[str] = None


class TimedUtterance(BaseModel):
    """Data model for an utterance with timing information"""

    speaker: str
    text: str
    start_seconds: float
    end_seconds: float


class TranscriptionCache:
    """Cache manager for transcription results to avoid repeated API calls"""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Use system temporary directory with a subdirectory for our cache
            temp_dir = Path(tempfile.gettempdir())
            cache_dir = temp_dir / "nlp_transcription_cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for the audio file based on content and metadata"""
        hasher = hashlib.sha256()

        # Include file size and modification time for quick check
        stat = file_path.stat()
        hasher.update(f"{file_path.name}:{stat.st_size}:{stat.st_mtime}".encode())

        # For small files, include content hash for accuracy
        if stat.st_size < 50 * 1024 * 1024:  # Less than 50MB
            with open(file_path, "rb") as f:
                # Read in chunks to handle larger files
                while chunk := f.read(8192):
                    hasher.update(chunk)

        return hasher.hexdigest()

    def _get_cache_path(self, file_hash: str, language: str) -> Path:
        """Get the cache file path for a given file hash and language"""
        return self.cache_dir / f"{file_hash}_{language}.json"

    def get_cached_response(self, audio_file: Path, language: str) -> Optional[dict]:
        """Retrieve cached transcription response if it exists"""
        try:
            file_hash = self._get_file_hash(audio_file)
            cache_path = self._get_cache_path(file_hash, language)

            if cache_path.exists():
                console.print(f"📋 Found cached transcription for {audio_file.name}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                return cached_data

        except Exception as e:
            logger.warning(f"Failed to retrieve cache: {e}")

        return None

    def save_response(self, audio_file: Path, language: str, response: dict) -> None:
        """Save transcription response to cache"""
        try:
            file_hash = self._get_file_hash(audio_file)
            cache_path = self._get_cache_path(file_hash, language)

            # Convert response to JSON-serializable format
            # The Deepgram response object needs to be converted to dict
            cache_data = self._response_to_dict(response)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            console.print(f"💾 Cached transcription for {audio_file.name}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _response_to_dict(self, response) -> dict:
        """Convert Deepgram response object to dictionary for JSON serialization"""
        try:
            # If response has a to_dict method, use it
            if hasattr(response, "to_dict"):
                return response.to_dict()

            # If it's already a dict, return as-is
            if isinstance(response, dict):
                return response

            # Try to convert response attributes to dict manually
            result = {}
            if hasattr(response, "results"):
                result["results"] = self._extract_results(response.results)

            return result

        except Exception as e:
            logger.warning(f"Failed to convert response to dict: {e}")
            # Fallback: try to serialize what we can
            return {
                "error": "Failed to serialize response",
                "type": str(type(response)),
            }

    def _extract_results(self, results):
        """Extract results from Deepgram response for caching"""
        try:
            results_dict = {}

            if hasattr(results, "channels"):
                results_dict["channels"] = []
                for channel in results.channels:
                    channel_dict = {}
                    if hasattr(channel, "alternatives"):
                        channel_dict["alternatives"] = []
                        for alt in channel.alternatives:
                            alt_dict = {}
                            if hasattr(alt, "transcript"):
                                alt_dict["transcript"] = alt.transcript
                            if hasattr(alt, "words"):
                                alt_dict["words"] = [
                                    {
                                        "word": getattr(word, "word", ""),
                                        "start": getattr(word, "start", 0),
                                        "end": getattr(word, "end", 0),
                                        "speaker": getattr(word, "speaker", None),
                                    }
                                    for word in alt.words
                                ]
                            channel_dict["alternatives"].append(alt_dict)
                    results_dict["channels"].append(channel_dict)

            if hasattr(results, "utterances"):
                results_dict["utterances"] = [
                    {
                        "transcript": getattr(utt, "transcript", ""),
                        "start": getattr(utt, "start", 0),
                        "end": getattr(utt, "end", 0),
                        "speaker": getattr(utt, "speaker", 0),
                    }
                    for utt in results.utterances
                ]

            return results_dict

        except Exception as e:
            logger.warning(f"Failed to extract results: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear all cached transcriptions"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            file_count = len(cache_files)

            for cache_file in cache_files:
                cache_file.unlink()

            if file_count > 0:
                console.print(f"🗑️ Cleared {file_count} transcription cache file(s)")
            else:
                console.print("🗑️ Cache was already empty")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


class DeepgramManager:
    """Manager for Deepgram speech-to-text operations"""

    def __init__(self):
        self.api_key = self._get_api_key()
        self.client = DeepgramClient(self.api_key)
        self.cache = TranscriptionCache()

    def _get_api_key(self) -> str:
        """Get Deepgram API key from environment"""
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            console.print("❌ DEEPGRAM_API_KEY environment variable not set!")
            console.print("💡 Get your free API key at: https://deepgram.com")
            console.print("💡 Set it with: export DEEPGRAM_API_KEY=your_key_here")
            raise typer.Exit(1)
        return api_key

    def transcribe_file(
        self,
        audio_file: Path,
        language: str = "en-US",
    ) -> Optional[dict]:
        """Transcribe audio file with speaker diarization using Deepgram"""

        # Check cache first
        cached_response = self.cache.get_cached_response(audio_file, language)
        if cached_response:
            return self._dict_to_response_object(cached_response)

        # If not cached, make API call
        try:
            console.print("🚀 Starting Deepgram diarization...")
            console.print(f"   File: {audio_file}")
            console.print(f"   Language: {language}")

            # Read the audio file
            with open(audio_file, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            # Configure options for diarization
            options = PrerecordedOptions(
                model="nova-3",
                language=language,
                smart_format=True,
                punctuate=True,
                diarize=True,
                utterances=True,
                paragraphs=True,
            )

            console.print("📤 Sending request to Deepgram...")
            start_time = time.time()

            # Make the transcription request
            response = self.client.listen.rest.v("1").transcribe_file(payload, options)

            elapsed = time.time() - start_time
            console.print(f"✅ Transcription completed in {elapsed:.1f}s!")

            # Cache the response
            self.cache.save_response(audio_file, language, response)

            return response

        except Exception as e:
            console.print(f"❌ Deepgram transcription failed: {e}")
            logger.error(f"Deepgram transcription failed: {e}")
            return None

    def _dict_to_response_object(self, cached_data: dict):
        """Convert cached dictionary back to a response-like object"""

        # Create a simple object that mimics the Deepgram response structure
        class MockResponse:
            def __init__(self, data):
                self.results = MockResults(data.get("results", {}))

        class MockResults:
            def __init__(self, results_data):
                self.channels = [
                    MockChannel(ch) for ch in results_data.get("channels", [])
                ]
                self.utterances = [
                    MockUtterance(utt) for utt in results_data.get("utterances", [])
                ]

        class MockChannel:
            def __init__(self, channel_data):
                self.alternatives = [
                    MockAlternative(alt) for alt in channel_data.get("alternatives", [])
                ]

        class MockAlternative:
            def __init__(self, alt_data):
                self.transcript = alt_data.get("transcript", "")
                self.words = [MockWord(word) for word in alt_data.get("words", [])]

        class MockWord:
            def __init__(self, word_data):
                self.word = word_data.get("word", "")
                self.start = word_data.get("start", 0)
                self.end = word_data.get("end", 0)
                self.speaker = word_data.get("speaker", None)

        class MockUtterance:
            def __init__(self, utt_data):
                self.transcript = utt_data.get("transcript", "")
                self.start = utt_data.get("start", 0)
                self.end = utt_data.get("end", 0)
                self.speaker = utt_data.get("speaker", 0)

        return MockResponse(cached_data)


class ConversationFormatter:
    """Formatter for diarized speech results"""

    @staticmethod
    def format_deepgram_response(response) -> Tuple[str, str, List[TimedUtterance]]:
        """Format Deepgram response into regular and diarized formats, plus timing data"""

        # Access response data directly - Deepgram SDK returns structured objects
        try:
            # The response is already a structured object with direct access to attributes
            if not hasattr(response, "results") or not response.results:
                console.print("❌ No results found in response")
                return "", "", []

            results = response.results
            if not hasattr(results, "channels") or not results.channels:
                console.print("❌ No channels found in results")
                return "", "", []

            channel = results.channels[0]
            if not hasattr(channel, "alternatives") or not channel.alternatives:
                console.print("❌ No alternatives found in channel")
                return "", "", []

            alternative = channel.alternatives[0]

            # Extract regular transcript
            regular_transcript = getattr(alternative, "transcript", "")

            # Extract diarized conversation from utterances (preferred) or words
            diarized_conversation = ""
            timed_utterances = []

            # Try utterances first (cleaner output)
            if hasattr(results, "utterances") and results.utterances:
                utterances = results.utterances
                console.print(
                    f"📝 Processing {len(utterances)} utterances for diarization"
                )

                # Group consecutive utterances by speaker
                conversation_turns = []
                current_speaker = None
                current_text_parts = []
                current_start = None
                current_end = None

                for utterance in utterances:
                    speaker_id = getattr(utterance, "speaker", 0)
                    text = getattr(utterance, "transcript", "").strip()
                    start = getattr(utterance, "start", 0)
                    end = getattr(utterance, "end", 0)

                    if not text:
                        continue

                    if current_speaker != speaker_id:
                        # Save previous speaker's text if exists
                        if current_speaker is not None and current_text_parts:
                            speaker_label = f"SPEAKER_{current_speaker:02d}"
                            combined_text = " ".join(current_text_parts).strip()
                            conversation_turns.append(
                                {
                                    "speaker": speaker_label,
                                    "text": combined_text,
                                    "start": current_start,
                                    "end": current_end,
                                }
                            )

                            # Add to timed utterances
                            timed_utterances.append(
                                TimedUtterance(
                                    speaker=speaker_label,
                                    text=combined_text,
                                    start_seconds=current_start,
                                    end_seconds=current_end,
                                )
                            )

                        # Start new speaker
                        current_speaker = speaker_id
                        current_text_parts = [text]
                        current_start = start
                        current_end = end
                    else:
                        # Same speaker, append text and update end time
                        current_text_parts.append(text)
                        current_end = end

                # Add the final speaker's text
                if current_speaker is not None and current_text_parts:
                    speaker_label = f"SPEAKER_{current_speaker:02d}"
                    combined_text = " ".join(current_text_parts).strip()
                    conversation_turns.append(
                        {
                            "speaker": speaker_label,
                            "text": combined_text,
                            "start": current_start,
                            "end": current_end,
                        }
                    )

                    # Add to timed utterances
                    timed_utterances.append(
                        TimedUtterance(
                            speaker=speaker_label,
                            text=combined_text,
                            start_seconds=current_start,
                            end_seconds=current_end,
                        )
                    )

                if conversation_turns:
                    diarized_conversation = "\n".join(
                        f"[{turn['speaker']}]: {turn['text']}"
                        for turn in conversation_turns
                    )

            # Fallback to words-based diarization if no utterances
            if (
                not diarized_conversation
                and hasattr(alternative, "words")
                and alternative.words
            ):
                words_info = alternative.words
                console.print(f"📝 Processing {len(words_info)} words for diarization")

                conversation_turns = ConversationFormatter._group_words_by_speaker(
                    words_info
                )

                if conversation_turns:
                    diarized_conversation = "\n".join(
                        f"[{turn['speaker']}]: {turn['text']}"
                        for turn in conversation_turns
                        if turn["text"]
                    )

            # Log statistics
            ConversationFormatter._log_statistics(
                diarized_conversation, regular_transcript
            )

            return regular_transcript.strip(), diarized_conversation, timed_utterances

        except Exception as e:
            console.print(f"❌ Error processing Deepgram response: {e}")
            logger.error(f"Error processing Deepgram response: {e}")
            return "", "", []

    @staticmethod
    def _group_words_by_speaker(words_info) -> List[dict]:
        """Group words by speaker"""
        conversation_turns = []
        current_speaker = None
        current_text = []

        for word_info in words_info:
            speaker_tag = getattr(word_info, "speaker", 0)
            word = getattr(word_info, "word", "")

            if current_speaker != speaker_tag:
                if current_speaker is not None and current_text:
                    conversation_turns.append(
                        {
                            "speaker": f"SPEAKER_{current_speaker:02d}",
                            "text": " ".join(current_text).strip(),
                        }
                    )
                current_speaker = speaker_tag
                current_text = [word]
            else:
                current_text.append(word)

        # Add the last speaker
        if current_speaker is not None and current_text:
            conversation_turns.append(
                {
                    "speaker": f"SPEAKER_{current_speaker:02d}",
                    "text": " ".join(current_text).strip(),
                }
            )

        return conversation_turns

    @staticmethod
    def _log_statistics(diarized_conversation: str, regular_transcript: str) -> None:
        """Log conversation statistics"""
        if diarized_conversation:
            lines = [line for line in diarized_conversation.split("\n") if line.strip()]
            unique_speakers = set()
            for line in lines:
                if line.startswith("[SPEAKER_"):
                    speaker = line.split("]:")[0] + "]"
                    unique_speakers.add(speaker)

            console.print(
                f"📝 Regular transcript: {len(regular_transcript)} characters"
            )
            console.print(
                f"👥 Detected {len(unique_speakers)} speakers: {', '.join(sorted(unique_speakers))}"
            )
            console.print(f"💬 Generated {len(lines)} conversation turns")


class ChapterGenerator:
    """Generator for creating chapters from transcripts using LLM"""

    def __init__(self):
        # Get the base model and add structured output
        base_llm = langchain_helper.get_model(google_flash=True)
        self.llm = base_llm.with_structured_output(ChapterResponse)

    def create_timed_transcript(self, timed_utterances: List[TimedUtterance]) -> str:
        """Create a timestamp-annotated transcript for LLM analysis"""
        lines = []
        for utterance in timed_utterances:
            # Format time as HH:MM:SS
            start_time = self._seconds_to_timestamp(utterance.start_seconds)
            lines.append(f"[{start_time}] [{utterance.speaker}]: {utterance.text}")

        return "\n".join(lines)

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def generate_chapters(
        self, timed_utterances: List[TimedUtterance]
    ) -> List[Chapter]:
        """Generate chapters using LLM analysis of the timed transcript"""
        if not timed_utterances:
            return []

        # Create timed transcript for LLM
        timed_transcript = self.create_timed_transcript(timed_utterances)
        total_duration = timed_utterances[-1].end_seconds

        # Create prompt for chapter generation
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert at analyzing conversations and creating meaningful chapter divisions.

Analyze the following timestamped conversation transcript and create chapters that represent natural topic transitions and conversation flow.

Guidelines:
- Create AT LEAST 3 chapters for any recording longer than 5 minutes
- For recordings under 5 minutes, create 2-3 chapters if there are natural breaks
- Each chapter should ideally be 2-5 minutes long (avoid very short chapters under 90 seconds unless necessary)
- Chapter titles should be descriptive and engaging, capturing the main topic or theme
- Use exact timestamps in seconds (not HH:MM:SS format)
- Look for natural conversation breaks: topic shifts, speaker transitions, Q&A sections, etc.
- Prefer chapters that capture complete thoughts or discussion segments
- Chapters must not overlap and should cover the entire duration
- First chapter should start at 0, last chapter should end at {duration}

Transcript:
{transcript}

Total Duration: {duration} seconds ({duration_minutes:.1f} minutes)

Please return a structured response with chapters covering the full conversation duration.
Include brief reasoning for your chapter divisions.
""")

        try:
            console.print("🤖 Generating chapters with LLM...")

            # Format the prompt
            formatted_prompt = prompt_template.format(
                transcript=timed_transcript,
                duration=int(total_duration),
                duration_minutes=total_duration / 60,
            )

            # Get structured LLM response
            response: ChapterResponse = self.llm.invoke(formatted_prompt)

            # Validate and fix chapter timing
            validated_chapters = self._validate_chapters(
                response.chapters, total_duration
            )

            console.print(f"✅ Generated {len(validated_chapters)} chapters")
            return validated_chapters

        except Exception as e:
            console.print(f"❌ Error generating chapters: {e}")
            logger.error(f"Error generating chapters: {e}")

            # Return fallback chapter
            return [
                Chapter(
                    title="Full Recording", start_seconds=0, end_seconds=total_duration
                )
            ]

    def _validate_chapters(
        self, chapters: List[Chapter], total_duration: float
    ) -> List[Chapter]:
        """Validate and fix chapter timing issues"""
        if not chapters:
            return [
                Chapter(
                    title="Full Recording", start_seconds=0, end_seconds=total_duration
                )
            ]

        # Sort chapters by start time
        chapters.sort(key=lambda c: c.start_seconds)

        validated = []
        for i, chapter in enumerate(chapters):
            # Ensure start time is valid
            start_time = max(0, chapter.start_seconds)

            # Ensure end time is valid and after start time
            end_time = min(total_duration, chapter.end_seconds)
            if end_time <= start_time:
                end_time = start_time + 30  # Minimum 30 seconds

            # Ensure no overlap with next chapter
            if i < len(chapters) - 1:
                next_start = chapters[i + 1].start_seconds
                if end_time > next_start:
                    end_time = next_start

            # Ensure last chapter ends at total duration
            if i == len(chapters) - 1:
                end_time = total_duration

            validated.append(
                Chapter(
                    title=chapter.title, start_seconds=start_time, end_seconds=end_time
                )
            )

        # Ensure first chapter starts at 0
        if validated and validated[0].start_seconds > 0:
            validated[0] = Chapter(
                title=validated[0].title,
                start_seconds=0,
                end_seconds=validated[0].end_seconds,
            )

        return validated


class ChapterManager:
    """Manager for applying chapters to audio files using ffmpeg"""

    @staticmethod
    def create_ffmpeg_metadata(chapters: List[Chapter]) -> str:
        """Create ffmpeg metadata format for chapters"""
        metadata_lines = [";FFMETADATA1"]

        for chapter in chapters:
            # FFmpeg expects timestamps in microseconds
            start_microseconds = int(chapter.start_seconds * 1_000_000)
            end_microseconds = int(chapter.end_seconds * 1_000_000)

            metadata_lines.extend(
                [
                    "",
                    "[CHAPTER]",
                    "TIMEBASE=1/1000000",
                    f"START={start_microseconds}",
                    f"END={end_microseconds}",
                    f"title={chapter.title}",
                ]
            )

        return "\n".join(metadata_lines)

    @staticmethod
    def apply_chapters_to_file(
        input_file: Path, output_file: Path, chapters: List[Chapter]
    ) -> bool:
        """Apply chapters to audio file using ffmpeg"""
        try:
            console.print("🎬 Applying chapters to audio file...")

            # Create temporary metadata file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                metadata_content = ChapterManager.create_ffmpeg_metadata(chapters)
                temp_file.write(metadata_content)
                temp_metadata_path = temp_file.name

            try:
                # Run ffmpeg command to add chapters
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_file),
                    "-i",
                    temp_metadata_path,
                    "-map_metadata",
                    "1",
                    "-codec",
                    "copy",
                    "-y",  # Overwrite output file if it exists
                    str(output_file),
                ]

                console.print(f"   Running: {' '.join(cmd)}")

                subprocess.run(cmd, capture_output=True, text=True, check=True)

                console.print("✅ Chapters applied successfully!")
                console.print(f"📄 Output saved to: {output_file}")
                return True

            finally:
                # Clean up temporary metadata file
                Path(temp_metadata_path).unlink(missing_ok=True)

        except subprocess.CalledProcessError as e:
            console.print(f"❌ FFmpeg failed: {e}")
            console.print(f"   stderr: {e.stderr}")
            logger.error(f"FFmpeg failed: {e}")
            return False
        except FileNotFoundError:
            console.print(
                "❌ FFmpeg not found. Please install FFmpeg to use chapter functionality."
            )
            console.print("💡 Install with: brew install ffmpeg")
            return False
        except Exception as e:
            console.print(f"❌ Error applying chapters: {e}")
            logger.error(f"Error applying chapters: {e}")
            return False

    @staticmethod
    def display_chapters(chapters: List[Chapter]) -> None:
        """Display chapters in a nice format"""
        console.print("\n🎬 Generated Chapters:")
        console.print("=" * 50)

        for i, chapter in enumerate(chapters, 1):
            start_time = ChapterManager._seconds_to_timestamp(chapter.start_seconds)
            end_time = ChapterManager._seconds_to_timestamp(chapter.end_seconds)
            duration = chapter.end_seconds - chapter.start_seconds

            console.print(f"Chapter {i}: {chapter.title}")
            console.print(f"   Time: {start_time} - {end_time} ({duration:.1f}s)")
            console.print()

    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_results(
    regular_transcript: str, diarized_conversation: str, output_file: Path
) -> None:
    """Save results to file"""
    full_output = ""
    if regular_transcript:
        full_output += "=== REGULAR TRANSCRIPT ===\n"
        full_output += regular_transcript + "\n\n"

    if diarized_conversation:
        full_output += "=== DIARIZED CONVERSATION ===\n"
        full_output += diarized_conversation + "\n"

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_output)

    console.print(f"✅ Results saved to: {output_file}")


@app.command(help="Transcribe audio file with speaker diarization")
def transcribe(
    audio_file: Annotated[Path, typer.Argument(help="Audio file to diarize")],
    output_file: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output text file")
    ] = None,
    language: Annotated[
        str, typer.Option("--language", help="Language code")
    ] = "en-US",
    show_results: Annotated[
        bool, typer.Option("--show", help="Show results in terminal")
    ] = False,
):
    """
    Transcribe audio file with speaker diarization using Deepgram.

    This command processes an audio file and generates both a regular transcript
    and a diarized conversation with speaker labels.
    """

    if not audio_file.exists():
        console.print(f"❌ Audio file not found: {audio_file}")
        raise typer.Exit(1)

    # Set default output file
    if output_file is None:
        output_file = audio_file.parent / f"{audio_file.stem}_diarized.txt"

    console.print(f"🎵 Processing: {audio_file}")
    console.print(f"📄 Output: {output_file}")

    # Initialize Deepgram manager
    deepgram = DeepgramManager()

    # Transcribe the file
    response = deepgram.transcribe_file(audio_file, language)

    if not response:
        console.print("❌ Transcription failed")
        raise typer.Exit(1)

    # Format the response
    regular_transcript, diarized_conversation, timed_utterances = (
        ConversationFormatter.format_deepgram_response(response)
    )

    if not regular_transcript and not diarized_conversation:
        console.print("❌ No transcript generated")
        raise typer.Exit(1)

    # Save results
    save_results(regular_transcript, diarized_conversation, output_file)

    # Show results if requested
    if show_results:
        console.print("\n" + "=" * 50)
        console.print("📄 RESULTS")
        console.print("=" * 50)

        if regular_transcript:
            console.print("\n📝 Regular Transcript:")
            console.print(
                regular_transcript[:500]
                + ("..." if len(regular_transcript) > 500 else "")
            )

        if diarized_conversation:
            console.print("\n👥 Diarized Conversation:")
            lines = diarized_conversation.split("\n")[:10]
            for line in lines:
                if line.strip():
                    console.print(f"   {line}")
            if len(diarized_conversation.split("\n")) > 10:
                console.print("   ...")

    console.print("\n🎯 Diarization completed successfully!")
    console.print("💡 Generate audio with different voices:")
    console.print(f"   uv run tts_dialog.py recreate {output_file}")


@app.command(help="Add chapters to audio file using AI-generated chapter markers")
def chapters(
    audio_file: Annotated[Path, typer.Argument(help="Audio file to add chapters to")],
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output audio file with chapters"),
    ] = None,
    language: Annotated[
        str, typer.Option("--language", help="Language code")
    ] = "en-US",
    show_chapters: Annotated[
        bool, typer.Option("--show", help="Show generated chapters")
    ] = False,
):
    """
    Add AI-generated chapters to an audio file.

    This command:
    1. Transcribes the audio with speaker diarization using Deepgram
    2. Uses an LLM to analyze the conversation and generate meaningful chapters
    3. Applies the chapters to the audio file using ffmpeg

    The output file will have embedded chapter markers that can be seen in media players.
    """

    if not audio_file.exists():
        console.print(f"❌ Audio file not found: {audio_file}")
        raise typer.Exit(1)

    # Set default output file
    if output_file is None:
        output_file = (
            audio_file.parent / f"{audio_file.stem}_with_chapters{audio_file.suffix}"
        )

    console.print(f"🎵 Processing: {audio_file}")
    console.print(f"📄 Output: {output_file}")

    # Step 1: Transcribe the file
    console.print("\n📝 Step 1: Transcribing audio...")
    deepgram = DeepgramManager()
    response = deepgram.transcribe_file(audio_file, language)

    if not response:
        console.print("❌ Transcription failed")
        raise typer.Exit(1)

    # Format the response and get timing data
    regular_transcript, diarized_conversation, timed_utterances = (
        ConversationFormatter.format_deepgram_response(response)
    )

    if not timed_utterances:
        console.print("❌ No transcript with timing data generated")
        raise typer.Exit(1)

    # Step 2: Generate chapters using LLM
    console.print("\n🤖 Step 2: Generating chapters...")
    chapter_generator = ChapterGenerator()
    chapters = chapter_generator.generate_chapters(timed_utterances)

    if not chapters:
        console.print("❌ No chapters generated")
        raise typer.Exit(1)

    # Show chapters if requested
    if show_chapters:
        ChapterManager.display_chapters(chapters)

    # Step 3: Apply chapters to audio file
    console.print("\n🎬 Step 3: Applying chapters to audio file...")
    success = ChapterManager.apply_chapters_to_file(audio_file, output_file, chapters)

    if not success:
        console.print("❌ Failed to apply chapters")
        raise typer.Exit(1)

    console.print("\n🎯 Chapter generation completed successfully!")
    console.print(f"🎵 Audio file with chapters: {output_file}")
    console.print("💡 Open the file in a media player to see the chapter markers!")


@app.command(help="Clear the transcription cache")
def clear_cache():
    """
    Clear all cached transcription results.

    Use this if you want to force re-transcription of files or if cache files are taking up too much space.
    """
    cache = TranscriptionCache()
    console.print(f"📁 Cache location: {cache.cache_dir}")
    cache.clear_cache()


if __name__ == "__main__":
    app()
