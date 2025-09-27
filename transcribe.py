#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "typer",
#     "loguru",
#     "rich",
#     "icecream",
#     "yt-dlp",
#     "assemblyai",
#     "langchain",
#     "langchain-core",
#     "langchain-openai",
#     "langchain-anthropic",
#     "langchain-google-genai",
#     "langchain-groq",
#     "openai",
#     "tiktoken",
#     "tenacity",
# ]
# ///

from datetime import datetime
import typer
import subprocess
import re
from typing import Optional, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

from loguru import logger
from rich import print
from pathlib import Path
from icecream import ic
import os
import assemblyai as aai
import langchain_helper
import openai_wrapper
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate


app = typer.Typer(no_args_is_help=True)


# ============================================================================
# Shared Helper Functions
# ============================================================================

def is_youtube_url(url: str) -> bool:
    """Check if a string is a YouTube URL"""
    youtube_patterns = [
        r"youtube\.com/watch\?v=",
        r"youtu\.be/",
        r"youtube\.com/embed/",
        r"youtube\.com/v/"
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def get_temp_dir() -> Path:
    """Get the temporary directory for downloads"""
    temp_dir = Path("/tmp/claude")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def save_transcript(transcript: str, prefix: str = "transcript") -> Path:
    """Save transcript to file and return the path"""
    output_file = get_temp_dir() / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_file.write_text(transcript)
    print(f"\nTranscript saved to: {output_file}")
    return output_file


# ============================================================================
# YouTube / Caption Processing (LangChain)
# ============================================================================

def fetch_youtube_captions(url: str) -> Optional[str]:
    """Fetch captions from YouTube video"""
    temp_dir = get_temp_dir()

    print("Fetching YouTube captions...")
    try:
        result = subprocess.run([
            "yt-dlp",
            "--write-auto-sub",
            "--skip-download",
            "--sub-format", "vtt",
            "-o", str(temp_dir / "%(title)s.%(ext)s"),
            url
        ], capture_output=True, text=True, timeout=60)  # 1 minute timeout

        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Caption download timed out after 1 minute")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching captions: {e}")
        return None

    # Find the VTT file
    vtt_files = list(temp_dir.glob("*.vtt"))
    if not vtt_files:
        logger.warning("No VTT caption files found")
        return None

    try:
        content = vtt_files[0].read_text()
        # Clean up the VTT file
        vtt_files[0].unlink()
        return content
    except Exception as e:
        logger.error(f"Error reading VTT file: {e}")
        # Try to clean up on error
        try:
            vtt_files[0].unlink()
        except Exception:
            pass
        return None


def chunk_text(text: str, max_tokens: int = 30_000) -> List[str]:
    """Split text into chunks that fit within token limits

    Default chunk size is 30k tokens to leave room for output with
    gpt-oss-120b's 65k output context window.
    """
    # Estimate 4 characters per token (rough estimate)
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return [text]

    # Split by double newlines first (natural breaks)
    chunks = []
    current_chunk = ""

    # Split by timestamps for VTT content
    lines = text.split('\n')
    for line in lines:
        if len(current_chunk) + len(line) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = line
        else:
            current_chunk += '\n' + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_llm_for_model(model_name: str):
    """Get the appropriate LLM based on model name"""
    model_map = {
        "claude": {"claude": True},
        "google": {"google": True},
        "llama": {"llama": True},
        "openai": {"openai": True},
        "openai_mini": {"openai_mini": True},
        "gpt_oss": {"gpt_oss": True},
        "grok4_fast": {"grok4_fast": True}  # Add grok-4-fast
    }

    params = model_map.get(model_name, {"grok4_fast": True})  # Default to grok4_fast
    return langchain_helper.get_model(**params)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def process_chunk_with_retry(chain, chunk_num: int) -> str:
    """Process a chunk with retry logic for rate limiting"""
    try:
        result = await chain.ainvoke({})
        return result
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            # Extract wait time if available
            match = re.search(r'try again in ([\d.]+)ms', str(e))
            if match:
                wait_ms = float(match.group(1))
                wait_seconds = wait_ms / 1_000.0 + 0.5  # Add buffer
                print(f"  Rate limit hit for chunk {chunk_num}. Waiting {wait_seconds:.1f}s...")
                await asyncio.sleep(wait_seconds)
            else:
                # Default wait if we can't parse the message
                print(f"  Rate limit hit for chunk {chunk_num}. Waiting 2s...")
                await asyncio.sleep(2)
        raise  # Re-raise to trigger tenacity retry


async def process_chunk_async(chunk_data: tuple, llm, prompt_template: str) -> tuple:
    """Process a single chunk asynchronously with retry logic"""
    chunk_num, total_chunks, chunk = chunk_data

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt_template +
            f"\n\nNOTE: This is chunk {chunk_num} of {total_chunks} from a longer video. "
            "Maintain continuity with previous chunks."),
        HumanMessage(content=chunk)
    ])

    chain = prompt | llm | StrOutputParser()

    # Use retry wrapper
    result = await process_chunk_with_retry(chain, chunk_num)
    return chunk_num, result


def process_youtube_with_llm(url: str, model: str, max_chunks: Optional[int] = None, single_chunk: bool = False) -> Optional[str]:
    """Process YouTube video using captions and LLM"""
    vtt_content = fetch_youtube_captions(url)

    if not vtt_content:
        print("No captions found for this video")
        return None

    llm = get_llm_for_model(model)

    # Check token count
    token_count = openai_wrapper.num_tokens_from_string(vtt_content)
    ic(f"Caption tokens: {token_count}")

    # Better prompt for transcript cleaning
    TRANSCRIPT_CLEANING_PROMPT = """You are an expert transcript editor. Convert these raw YouTube captions into a clean, readable transcript.

Your task:
1. Remove all timestamps and technical formatting (WEBVTT headers, etc.)
2. Fix caption errors, typos, and mis-transcriptions
3. Add proper punctuation and paragraph breaks
4. Identify and label different speakers (e.g., "Host:", "Guest:", "Speaker 1:")
5. Remove duplicate text that appears from caption overlap
6. Fix sentence fragments and incomplete thoughts
7. Preserve the actual content - do not summarize or skip anything
8. Format dialogue naturally with speaker changes on new lines

IMPORTANT: Preserve all content except obvious errors. This is a transcription, not a summary."""

    if token_count > 100_000 and not single_chunk:  # Very long video
        print(f"\nWarning: This is a very long video ({token_count} tokens).")
        print("Processing in chunks...")

        chunks = chunk_text(vtt_content, max_tokens=30_000)  # Larger chunks for gpt-oss-120b
        print(f"Split into {len(chunks)} chunks for processing")

        # Apply max_chunks limit if specified
        if max_chunks and max_chunks < len(chunks):
            chunks = chunks[:max_chunks]
            print(f"Processing only first {max_chunks} chunks (testing mode)")

        # Process all chunks at once without concurrency limits
        print(f"Processing {len(chunks)} chunks all at once...")

        # Create async event loop for parallel processing
        async def process_all_chunks():
            # Prepare chunk data with indices
            chunk_data = [(i, len(chunks), chunk) for i, chunk in enumerate(chunks, 1)]

            # Process all chunks simultaneously without semaphore
            print(f"Starting all {len(chunks)} chunks simultaneously...")
            tasks = [process_chunk_async(data, llm, TRANSCRIPT_CLEANING_PROMPT) for data in chunk_data]
            results = await asyncio.gather(*tasks)

            # Sort results by chunk number to maintain order
            results.sort(key=lambda x: x[0])
            print(f"Completed all {len(chunks)} chunks")
            return [result[1] for result in results]

        # Run the async processing
        full_transcript = asyncio.run(process_all_chunks())
        return "\n\n".join(full_transcript)
    else:
        # Process in one go for shorter videos or when single_chunk is True
        if single_chunk:
            print(f"Processing entire transcript as single chunk ({token_count} tokens)...")
            print(f"Caption content length: {len(vtt_content)} characters")

        start_time = time.time()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=TRANSCRIPT_CLEANING_PROMPT),
            HumanMessage(content=vtt_content)
        ])

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Output length: {len(result)} characters")
        output_tokens = openai_wrapper.num_tokens_from_string(result)
        print(f"Output tokens: {output_tokens}")

        return result


# ============================================================================
# Audio File Processing (AssemblyAI)
# ============================================================================

def download_youtube_audio(youtube_url: str) -> Optional[Path]:
    """Download audio from YouTube video using yt-dlp"""
    temp_dir = get_temp_dir()
    output_path = temp_dir / f"youtube_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

    try:
        result = subprocess.run([
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "-o", str(output_path),
            youtube_url
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode != 0:
            logger.error(f"Failed to download audio: {result.stderr}")
            return None

        print(f"Downloaded audio to: {output_path}")
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("Audio download timed out after 5 minutes")
        # Clean up partial download if it exists
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading audio: {e}")
        # Clean up partial download if it exists
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return None


def process_audio_with_assemblyai(audio_source, diarization: bool = True, srt: bool = False) -> Optional[str]:
    """Process audio file using AssemblyAI"""
    aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    if not aai.settings.api_key:
        print("Please set ASSEMBLYAI_API_KEY environment variable")
        print("Get your API key from: https://www.assemblyai.com/")
        return None

    config = aai.TranscriptionConfig(speaker_labels=diarization)

    print(f"Transcribing audio using AssemblyAI...")
    ic(audio_source, datetime.now())

    transcript = aai.Transcriber().transcribe(audio_source, config)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed: {transcript.error}")
        return None

    # Format the output
    output = []

    if transcript.utterances:
        for utterance in transcript.utterances:
            output.append(f"Speaker {utterance.speaker}: {utterance.text}")
    else:
        output.append(transcript.text or "No transcript text found")

    full_transcript = "\n\n".join(output)

    if srt:
        print("\n=== SRT Format ===")
        print(transcript.export_subtitles_srt())

    return full_transcript


# ============================================================================
# Main Command
# ============================================================================

@app.command()
def transcribe(
    path: str = typer.Argument(None, help="File path or YouTube URL to transcribe"),
    diarization: bool = typer.Option(True, help="Enable speaker diarization (AssemblyAI only)"),
    srt: bool = typer.Option(False, help="Export SRT subtitles (AssemblyAI only)"),
    model: str = typer.Option("grok4_fast", help="Model for YouTube captions: grok4_fast, gpt_oss, openai, claude, google, llama, openai_mini"),
    max_chunks: int = typer.Option(None, help="Maximum number of chunks to process (for testing)"),
    use_assemblyai_for_youtube: bool = typer.Option(False, help="Download audio and use AssemblyAI for YouTube URLs"),
    single_chunk: bool = typer.Option(False, help="Process entire transcript as a single chunk (no splitting)")
):
    """
    Transcribe audio from files or YouTube videos.

    - For YouTube URLs: Uses captions + LLM by default (or AssemblyAI if specified)
    - For audio files: Uses AssemblyAI
    """
    if not path:
        print("Please provide either a file path or a YouTube URL")
        return

    transcript = None
    is_youtube = is_youtube_url(path)

    if is_youtube:
        print(f"Detected YouTube URL: {path}")

        if use_assemblyai_for_youtube:
            # Download audio and use AssemblyAI
            print("Downloading audio for AssemblyAI processing...")
            audio_file = download_youtube_audio(path)
            if audio_file:
                try:
                    with open(audio_file, "rb") as f:
                        transcript = process_audio_with_assemblyai(f, diarization, srt)
                finally:
                    # Always clean up downloaded file
                    if audio_file.exists():
                        audio_file.unlink()
                        print(f"Cleaned up temporary file: {audio_file}")
        else:
            # Use captions + LLM (default for YouTube)
            print(f"Using {model} model with YouTube captions")
            transcript = process_youtube_with_llm(path, model, max_chunks, single_chunk)
    else:
        # Local audio file - use AssemblyAI
        audio_file = Path(path)
        if not audio_file.exists():
            print(f"File not found: {audio_file}")
            return

        with open(audio_file, "rb") as f:
            transcript = process_audio_with_assemblyai(f, diarization, srt)

    if transcript:
        print("\n=== Transcript ===")
        print(transcript)
        save_transcript(transcript)
    else:
        print("Transcription failed or no output generated")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()