#!python3

import typer
from loguru import logger
from rich.console import Console
import openai_wrapper
import ell
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option
import os

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()


def show(img):
    if "TMUX" in os.environ:
        os.system(f"timg --grid 4 --title -ps {img}")
    else:
        os.system(f"timg --grid 4 --title -p {img}")


@app.command()
def studio(port: int = Option(None, help="Port to run the ELL Studio on")):
    """
    Launch the ELL Studio interface for interactive model exploration and testing.

    This command opens the ELL Studio, allowing users to interactively work with
    language models, test prompts, and analyze responses in a user-friendly environment.
    """
    run_studio(port=port)


@ell.simple(model=get_ell_model(openai=True), n=2)
def prompt_illustrate(content: str):
    """
    Create an image to represent this blog post. It should be fun whimsical and if it features people make them a 3d render of a cartoon racoon in the pixar style, the main racoon should be in his late 30s wearing very colorful glasses, and whimsical.
    """
    return content  # This will be the user prompt


# https://docs.ell.so/core_concepts/multimodality.html
# Dalle-3 isn't implemented yet :(


@app.command()
def draw(
    path: str = typer.Argument(None),
):
    import image

    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    ret = prompt_illustrate(user_text)
    for r in ret:
        image.gen_igor(str(r))


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
