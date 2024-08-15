#!python3

import sys
import asyncio

from langchain_core import messages


import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich.console import Console
import langchain_helper
from icecream import ic
import openai_wrapper

console = Console()
app = typer.Typer(no_args_is_help=True)


@logger.catch()
def app_wrap_loguru():
    app()


def prompt_fix(user_text):
    instructions = """You are an advanced AI with superior spelling correction abilities.
Your task is to correct any spelling errors you encounter in the text provided below.
The text is markdown, don't change the markdown formatting.
Don't add a markdown block around the entire text
Don't change any meanings
Fix all the supplied text
Do nt change wrapping
    """
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=user_text),
        ]
    )


@app.command()
def fix(
    trace: bool = False,
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
):
    langchain_helper.langsmith_trace_if_requested(
        trace,
        lambda: asyncio.run(
            inner_fix(openai=openai, google=google, claude=claude, llama=llama)
        ),
    )


async def inner_fix(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
):
    from datetime import datetime

    start = datetime.now()
    # Use a mid speed model
    llm = langchain_helper.get_model(
        openai=openai, google=google, claude=claude, llama=llama
    )
    ic(langchain_helper.get_model_name(llm))

    user_text = "".join(sys.stdin.readlines())
    fixed = await (prompt_fix(user_text) | llm).ainvoke({})
    elapsed_seconds = (datetime.now() - start).total_seconds()
    ic(int(elapsed_seconds))
    ic(len(user_text), len(fixed.content))
    ic(openai_wrapper.num_tokens_from_string(user_text))
    print("--erase me--")

    print(fixed.content)


if __name__ == "__main__":
    app_wrap_loguru()
