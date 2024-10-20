#!python3
# pylint: disable=missing-function-docstring

import typer
from rich.console import Console
import ell
import os
import openai_wrapper
from loguru import logger

console = Console()
app = typer.Typer(no_args_is_help=True)

# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)


@ell.simple(model=openai_wrapper.get_ell_model(openai=True))
def prompt_captions_to_human_readable(captions: str, last_chunk: str):
    """
    You are a super smart AI, who understands captions formats, and also English grammar and spelling.

    You will be given captions as input, you should output the text that can be read by a human like a book.

    Be sure to include punctuation, correct errors and include paragraphs when they make sense.

    Even if the person is talking continuously, still make sentences and paragraphs.

    A sentence should not be more than 30 words, and a paragraph should not be more than 10 sentences.

    If you think we're in an add, put it into <sponsor> </sponsor> links

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
    chunk_size = 20_000
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

    @ell.complex(model=openai_wrapper.get_ell_model(openai=True))
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
