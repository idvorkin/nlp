#!python3


import sys
import asyncio

from langchain_core import messages

import typer
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

## Instructions
* Start with a commit 1 line summary. Be concise, but informative.
* Then add details,
* If you see bugs, start at the top of the file with
* When listing  changes,
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
* If you see any of the following, skip them, or at most list them last as a single line
    * Changes to formatting/whitespace
    * Changes to imports
    * Changes to comments

Example:
Descriptive message
### BUGS: (Only include if bugs are seen)
    * List bugs
### Reason for change
    * reason
### Details
    * details
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
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_build_commit())
    )


async def a_build_commit():
    llms = [
        langchain_helper.get_model(openai=True),
        langchain_helper.get_model(claude=True),
    ]
    # google = langchain_helper.get_model(google=True)

    async def describe_diff(user_text, llm):
        description = await (prompt_summarize_diff(user_text) | llm).ainvoke({})
        return description.content, llm

    user_text = "".join(sys.stdin.readlines())
    describe_diff_tasks = [describe_diff(user_text, llm) for llm in llms]
    describe_diffs = [result for result in await asyncio.gather(*describe_diff_tasks)]

    for description, llm in describe_diffs:
        print(f"# -- model: {langchain_helper.get_model_name(llm)} --")
        print(description)


if __name__ == "__main__":
    app_wrap_loguru()