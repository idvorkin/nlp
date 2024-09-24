#!python3

import sys
import typer
from loguru import logger
from rich.console import Console
from icecream import ic
import openai_wrapper
import ell
import os
from datetime import datetime

console = Console()
app = typer.Typer(no_args_is_help=True)

# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)


@logger.catch()
def app_wrap_loguru():
    app()


@ell.simple(model="gpt-4o")
def prompt_fix(user_text: str):
    """You are an advanced AI with superior spelling correction abilities.
    Your task is to correct any spelling errors you encounter in the text provided below.
    The text is markdown, don't change the markdown formatting.
    Don't add a markdown block around the entire text
    Don't change any meanings
    Fix all the supplied text
    Do not change wrapping
    """
    return user_text  # This will be the user prompt


@app.command()
def fix():
    start = datetime.now()

    user_text = "".join(sys.stdin.readlines())
    fixed = prompt_fix(user_text)
    elapsed_seconds = (datetime.now() - start).total_seconds()
    ic(int(elapsed_seconds))
    ic(len(user_text), len(str(fixed)))
    ic(openai_wrapper.num_tokens_from_string(user_text))
    print("--erase me--")

    print(fixed)


if __name__ == "__main__":
    app_wrap_loguru()
