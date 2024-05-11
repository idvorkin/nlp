#!python3


from pathlib import Path
import sys
import asyncio
from langchain.schema.output_parser import StrOutputParser

from langchain_core import messages
import requests

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
import html2text


def prompt_illustrate(content):
    instructions = """
Help me create an image to represent this blog post, by making some prompts for GPT. They should be fun whimsical and if they feature someone that person is a 3d render of a cartoon racoon in the pixar style, the raccon can be in his mid 30s and wear glasses
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )


def get_text(path):
    if not path:  # read from stdin
        return "".join(sys.stdin.readlines())
    # check if path is URL
    if path.startswith("http"):
        request = requests.get(path)
        out = html2text.html2text(request.text)
        return out
    if path:
        # try to open the file, using pathlib
        return Path(path).read_text()
    # read stdin
    return str(sys.stdin.readlines())


async def a_draw(path: str):
    llm = langchain_helper.get_model(claude=True)
    user_text = get_text(path)
    ret = (prompt_illustrate(user_text) | llm | StrOutputParser()).invoke({})
    print(ret)


console = Console()
app = typer.Typer()


@app.command()
def draw(
    trace: bool = False,
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_draw(path))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
