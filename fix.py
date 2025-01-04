#!python3

import sys
import typer
from loguru import logger
from rich.console import Console
from icecream import ic
import openai_wrapper
import ell
from datetime import datetime
from pathlib import Path
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()


@logger.catch()
def app_wrap_loguru():
    app()


@ell.simple(model=get_ell_model(openai=True))
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
def fix(
    studio: bool = Option(False, help="Launch the ELL Studio interface"),
    port: int = Option(
        None, help="Port to run the ELL Studio on (only used with --studio)"
    ),
    file: Path = Option(None, help="Path to the file to be fixed"),
):
    """
    Fix text or launch the ELL Studio interface.

    This command allows you to fix text or open the ELL Studio for
    interactive model exploration and testing.
    """
    if studio:
        run_studio(port=port)
        return

    start = datetime.now()

    if file:
        user_text = file.read_text()
    else:
        user_text = "".join(sys.stdin.readlines())
    fixed = prompt_fix(user_text)
    elapsed_seconds = (datetime.now() - start).total_seconds()
    ic(int(elapsed_seconds))
    ic(len(user_text), len(str(fixed)))
    ic(openai_wrapper.num_tokens_from_string(user_text))
    if file:
        file.write_text(fixed)
    else:
        print(fixed)


if __name__ == "__main__":
    app_wrap_loguru()
