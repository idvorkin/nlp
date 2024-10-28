#!python3
# pylint: disable=missing-function-docstring

import typer
from rich.console import Console
import ell
from loguru import logger
from icecream import ic
from ell_helper import init_ell, run_studio, get_ell_model, to_gist
from typer import Option
import openai_wrapper
from pathlib import Path

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()


@app.command()
def studio(port: int = Option(None, help="Port to run the ELL Studio on")):
    """
    Launch the ELL Studio interface for interactive model exploration and testing.

    This command opens the ELL Studio, allowing users to interactively work with
    language models, test prompts, and analyze responses in a user-friendly environment.
    """
    run_studio(port=port)


# Use the cheap model as this is an easy task we put a lot of text through.
@ell.complex(model=get_ell_model(openai=True))
def prompt_captions_to_human_readable(captions: str, last_chunk: str, youtube_url: str):
    system = f"""You are an expert at converting raw video captions into clean, readable text. Your task is to:

1. Convert VTT captions into properly formatted markdown text
2. Fix grammar, spelling, and punctuation
3. Break text into logical paragraphs (max 10 sentences per paragraph)
4. Keep sentences under 30 words
5. Add speaker labels like "**HOST:**" or "**GUEST1:**" when speakers change, do not use a speaker label if the speaker is the same as the last speaker
6. Insert timecodes in this format: [00:01:34]({youtube_url}&t=94s)
   - Add them on a speaker change if we haven't had a timecode in the last 2 minutes
   - Put them on their own line
   - For timestamp 00:06:09.500, output [00:06:09]({youtube_url}&t=369s)
7. Replace ad/sponsor segments with "**Ad break**" on its own line
8. Never summarize or condense the content (except adbreaks)
9. Process only the new captions, not the previous chunk

Previous chunk for context (do not include in output):
<lastchunk>
{last_chunk}
</lastchunk>

Format the output as clean markdown text with proper paragraphs, punctuation and speaker labels.
"""
    return [ell.system(system), ell.user(captions)]  # type: ignore


def get_canonic_youtube_url(path: str) -> str:
    """Extract and normalize a YouTube URL from a path string.

    Handles various YouTube URL formats including:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - youtube.com/v/VIDEO_ID
    - youtube.com/embed/VIDEO_ID

    Returns:
        Canonical form: https://youtube.com/watch?v=VIDEO_ID
        Empty string if no YouTube URL found
    """
    if not path or not isinstance(path, str):
        return ""

    import re

    # Match common YouTube URL patterns
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            video_id = match.group(1)
            return f"https://youtube.com/watch?v={video_id}"

    return ""


@app.command()
def to_human(path: str = typer.Argument(None), gist: bool = True):
    """
    Process captions from standard input and output formatted text.

    This command reads captions from stdin, processes them in chunks,
    and prints the formatted output. It uses AI to improve readability
    and structure of the captions.
    """

    path = get_canonic_youtube_url(path)
    if not path:
        raise ValueError("No YouTube URL found in the provided path")
    youtube_url = path
    header = f"""
*Transcribed [{path}]({path}) via [transcribe.py](https://github.com/idvorkin/nlp/blob/main/transcribe.py)*
"""
    user_text = openai_wrapper.get_text_from_path_or_stdin(youtube_url)
    ic(header)
    output_text = header + "\n"

    last_chunk = ""
    for i, chunk in enumerate(split_string(user_text), 0):
        tokens = openai_wrapper.num_tokens_from_string(chunk)
        ic(i, tokens)

        response = prompt_captions_to_human_readable(chunk, last_chunk, youtube_url)
        response = response.content[0].text
        last_chunk = response
        output_text += "\n" + response

    output_path = Path(
        "~/tmp/human_captions.md"
    ).expanduser()  # get smarter about naming these.
    output_path.write_text(output_text)
    ic(output_path)
    if gist:
        # create temp file and write print buffer to it
        to_gist(output_path)
    else:
        print(output_text)


def split_string(input_string):
    # TODO: update this to use the tokenizer
    # Output size is 16K, so let that be the chunk size
    # A very rough estimate might suggest that a VTT file could be around 1.5 to 3 times larger than the plain text, depending on how verbose the timestamps and metadata are compared to the actual dialogue or text content. However, this is just an approximation and can vary significantly based on the specific content and structure of the VTT file.
    chars_per_token = 4
    max_tokens = 4_000
    vtt_multiplier = 2
    chunk_size = chars_per_token * max_tokens * vtt_multiplier
    ic(int(len(input_string) / chunk_size))
    for i in range(0, len(input_string), chunk_size):
        chunk = input_string[i : i + chunk_size]
        yield chunk


@app.command()
def captions_fix():
    """
    Fix and enhance captions from standard input.

    This command reads captions from stdin, processes them using AI to fix
    typos, remove filler words, suggest chapter summaries, and create titles.
    It outputs the enhanced captions along with additional metadata.
    """
    user_text = "".join(typer.get_text_stream("stdin").readlines())

    @ell.complex(model=get_ell_model(openai=True))
    def fix_captions(transcript: str):
        """
        You are an AI expert at fixing up captions files.

        Given this transcript, fix it up:

        E.g.

        <captions-fixed-up>

        Input Transcript cleaned up without typos, without uhms/ahs

        </captions-fixed-up>

        List words that were hard to translate

        <trouble_words>
        AED - From context hard to know what this is
        </trouble_words>

        Then suggest where to make chapter summaries for the Youtube description. Only include mm:ss

        <chapters>
        00:00 My description here
        00:10 My second description here
        </chapters>

        Also include nice titles

        <titles>
        1: Title 1
        2: Title 2
        </titles>
        """
        return [ell.user(transcript)]

    response = fix_captions(user_text)
    print(response)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
