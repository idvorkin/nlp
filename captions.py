#!python3
# pylint: disable=missing-function-docstring

import typer
from rich.console import Console
import ell
from loguru import logger
from icecream import ic
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option

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
@ell.simple(model=get_ell_model(openai_cheap=True))
def prompt_captions_to_human_readable(captions: str, last_chunk: str):
    """
    You are a super smart AI, who understands captions formats, and also English grammar and spelling.

    You will be given captions as input, you should output the text that can be read by a human like a book.

    Be sure to include punctuation, correct errors and include paragraphs when they make sense.

    Even if the person is talking continuously, still make sentences and paragraphs.

    A sentence should not be more than 30 words, and a paragraph should not be more than 10 sentences.

    The podcast has ads/sponsors. I don't want to see them. If you see one, don't include it. Just say "**Ad break**" in a new paragraph and skip the transcribe


    If you can figure out who is speaking start his paragraphs with "**HOST:**" or  "**GUEST1:**" etc.

    Because the podcast is long, you are looking 1 chunk at at a time. The last chunk before the one you are proecssing is
    <lastchunk>
    {last_chunk}
    </lastchunk>

    Do not output the last chunk, it's just htere to give you context in the chunk you are processing.

    """
    return captions


@app.command()
def to_human():
    """
    Process captions from standard input and output formatted text.

    This command reads captions from stdin, processes them in chunks,
    and prints the formatted output. It uses AI to improve readability
    and structure of the captions.
    """
    user_text = "".join(typer.get_text_stream("stdin").readlines())

    last_chunk = ""
    for chunk in split_string(user_text):
        response = prompt_captions_to_human_readable(chunk, last_chunk)
        last_chunk = chunk
        print(response)


def split_string(input_string):
    # TODO: update this to use the tokenizer
    # Output size is 16K, so let that be the chunk size
    # A very rough estimate might suggest that a VTT file could be around 1.5 to 3 times larger than the plain text, depending on how verbose the timestamps and metadata are compared to the actual dialogue or text content. However, this is just an approximation and can vary significantly based on the specific content and structure of the VTT file.
    chars_per_token = 4
    max_tokens = 15_000
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
