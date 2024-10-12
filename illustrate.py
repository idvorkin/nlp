#!python3

import typer
from loguru import logger
from rich import print
from rich.console import Console
import openai_wrapper
import ell
import os

# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)

console = Console()
app = typer.Typer(no_args_is_help=True)


@ell.simple(model=openai_wrapper.gpt4.name)
def prompt_illustrate(content: str):
    """
    Help me create an image to represent this blog post, by making some prompts for GPT. They should be fun whimsical and if they feature people but as 3d render of a cartoon racoon in the pixar style, the main racoon should be in his late 30s wearing very colorful glasses, and whimsical. End the prompt asking to render each photo, not just the first one
    """
    return content  # This will be the user prompt


# https://docs.ell.so/core_concepts/multimodality.html
# Dalle-3 isn't implemented yet :(


@app.command()
def draw(
    path: str = typer.Argument(None),
):
    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    ret = prompt_illustrate(user_text)
    print(ret)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
