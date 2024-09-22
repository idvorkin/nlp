#!python3
# pylint: disable=missing-function-docstring


import typer
from icecream import ic
from rich.console import Console
from loguru import logger
import ell
import os

app = typer.Typer()

# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)


@ell.simple(model="gpt-4o-mini")
def hello(world: str):
    """You are a unhelpful assistant"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def scratch():
    response = hello("Igor", lm_parms=dict(n=2))
    ic(response)


@app.command()
def studio():
    import subprocess

    subprocess.run(["ell-studio", "--storage", ELL_LOGDIR])


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
