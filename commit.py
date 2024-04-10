#!python3


import sys

from langchain_core import messages

import typer
from icecream import ic
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper

console = Console()
app = typer.Typer()


@logger.catch()
def app_wrap_loguru():
    app()


def prompt_summarize_diff(diff_output):
    instructions = """
You are an expert programmer, write a descriptive and informative commit message for a recent code change, which is presented as the output of git diff --staged.

Start with a commit 1 line summary. Be concise, but informative.
Then add details,
When listing  changes,
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_output),
        ]
    )


@app.command()
def build_commit(
    trace: bool = False,
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
):
    user_text = "".join(sys.stdin.readlines())

    def do_work():
        llm = langchain_helper.get_model(openai=openai, google=google, claude=claude)
        prompt_summarize_diff(user_text)
        output = (prompt_summarize_diff(user_text) | llm).invoke({}).content
        print(output)

    if not trace:
        return do_work()

    langchain_helper.langsmith_trace(do_work)


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
