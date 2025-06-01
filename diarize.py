#!uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer[all]",
#     "rich",
#     "deepgram-sdk",
#     "pydantic",
#     "icecream",
#     "loguru",
# ]
# ///

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from typing_extensions import Annotated
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from loguru import logger
from rich.console import Console

console = Console()
app = typer.Typer(
    name="diarize",
    help="Convert audio files to structured conversations using speaker diarization with Deepgram",
    add_completion=False,
    no_args_is_help=True,
)


class DeepgramManager:
    """Manager for Deepgram speech-to-text operations"""

    def __init__(self):
        self.api_key = self._get_api_key()
        self.client = DeepgramClient(self.api_key)

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
            return response

        except Exception as e:
            console.print(f"❌ Deepgram transcription failed: {e}")
            logger.error(f"Deepgram transcription failed: {e}")
            return None


class ConversationFormatter:
    """Formatter for diarized speech results"""

    @staticmethod
    def format_deepgram_response(response) -> Tuple[str, str]:
        """Format Deepgram response into regular and diarized formats"""

        # Access response data directly - Deepgram SDK returns structured objects
        try:
            # The response is already a structured object with direct access to attributes
            if not hasattr(response, "results") or not response.results:
                console.print("❌ No results found in response")
                return "", ""

            results = response.results
            if not hasattr(results, "channels") or not results.channels:
                console.print("❌ No channels found in results")
                return "", ""

            channel = results.channels[0]
            if not hasattr(channel, "alternatives") or not channel.alternatives:
                console.print("❌ No alternatives found in channel")
                return "", ""

            alternative = channel.alternatives[0]

            # Extract regular transcript
            regular_transcript = getattr(alternative, "transcript", "")

            # Extract diarized conversation from utterances (preferred) or words
            diarized_conversation = ""

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

                for utterance in utterances:
                    speaker_id = getattr(utterance, "speaker", 0)
                    text = getattr(utterance, "transcript", "").strip()

                    if not text:
                        continue

                    if current_speaker != speaker_id:
                        # Save previous speaker's text if exists
                        if current_speaker is not None and current_text_parts:
                            conversation_turns.append(
                                {
                                    "speaker": f"SPEAKER_{current_speaker:02d}",
                                    "text": " ".join(current_text_parts).strip(),
                                }
                            )

                        # Start new speaker
                        current_speaker = speaker_id
                        current_text_parts = [text]
                    else:
                        # Same speaker, append text
                        current_text_parts.append(text)

                # Add the final speaker's text
                if current_speaker is not None and current_text_parts:
                    conversation_turns.append(
                        {
                            "speaker": f"SPEAKER_{current_speaker:02d}",
                            "text": " ".join(current_text_parts).strip(),
                        }
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

            return regular_transcript.strip(), diarized_conversation

        except Exception as e:
            console.print(f"❌ Error processing Deepgram response: {e}")
            logger.error(f"Error processing Deepgram response: {e}")
            return "", ""

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
    regular_transcript, diarized_conversation = (
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


if __name__ == "__main__":
    app()
