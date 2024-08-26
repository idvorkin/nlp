#!python3


import asyncio
from langchain.schema.output_parser import StrOutputParser

from langchain_core import messages
import typer
from langchain.prompts import ChatPromptTemplate
from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper


def prompt_illustrate(content):
    instructions = """
Help me create an image to represent this blog post, by making some prompts for GPT. They should be fun whimsical and if they feature peope but as 3d render of a cartoon racoon in the pixar style, the main raccon should be in his late 30s wearing very colorful glasses, and whimsical. End the prompt asking to render each photo, not just  the first one
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )


async def a_draw(path: str):
    llm = langchain_helper.get_model(openai=True)
    user_text = langchain_helper.get_text_from_path_or_stdin(path)
    ret = (prompt_illustrate(user_text) | llm | StrOutputParser()).invoke({})
    print(ret)


console = Console()
app = typer.Typer(no_args_is_help=True)


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
