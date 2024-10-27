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
def prompt_captions_to_human_readable(captions: str, last_chunk: str):
    system = f"""
        You are a super smart AI who understands caption formats, English grammar, and spelling.

        You will be given captions as input. Output the text in a human-readable format, like a book.

        **The output should be verbatim. Do not summarize or condense text.**

        Include punctuation, correct errors, and add paragraphs where appropriate.

        To help users, we emit jumpable time codes like [00:01:00] based on the input time codes.  It should be on it's own line, ideally when switching from guest to host, representing the time code of the discussion in the original input so the user can find that time code. The timecode should be a markdown link e.g. [00:06:00](https://mypodcast/play/xxx?t=360s)

        Even if a person is talking continuously, break the text into sentences and paragraphs.

        Limit sentences to 30 words and paragraphs to 10 sentences.

        For ads/sponsors, skip the transcription and insert "**Ad break**" in a new paragraph.

        If you can identify speakers, start their paragraphs with "**HOST:**" or "**GUEST1:**", etc.

        You are processing the podcast in chunks. The last chunk before the one you are processing is:
        <lastchunk>
        {last_chunk}
        </lastchunk>

        Do not output the last chunk; it's provided for context in the chunk you are processing.
        """
    return [ell.system(system), ell.user(captions)]  # type: ignore


@app.command()
def to_human(path: str = typer.Argument(None), gist: bool = True):
    """
    Process captions from standard input and output formatted text.

    This command reads captions from stdin, processes them in chunks,
    and prints the formatted output. It uses AI to improve readability
    and structure of the captions.
    """

    header = f"""
*Transcribed [{path}]({path}) via [transcribe.py](https://github.com/idvorkin/nlp/blob/main/transcribe.py)*
"""
    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    ic(header)
    output_text = header + "\n"

    last_chunk = ""
    for i, chunk in enumerate(split_string(user_text), 0):
        tokens = openai_wrapper.num_tokens_from_string(chunk)
        ic(i, tokens)
        response = prompt_captions_to_human_readable(chunk, last_chunk)
        response = response.content[0].text
        last_chunk = response
        output_text += "\n" + response
        if i > 2:
            ic("BREAKING EARLY WHILE DEBUGGING")
            break
    # TODO Add gist support later

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
    max_tokens = 2_000
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
