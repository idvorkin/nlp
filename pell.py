#!python3

import typer
from icecream import ic
from rich.console import Console
from loguru import logger
import ell

app = typer.Typer()


@ell.simple(model="gpt-4o-mini")
def hello(world: str):
    """You are a helpful assistant"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def simple_1():
    response = hello("sam altman")  # just a str, "Hello Sam Altman! ..."
    ic(response)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
