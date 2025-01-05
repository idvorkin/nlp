#!python3


import sys
import asyncio

from langchain_core import messages
from langchain_core.language_models.chat_models import BaseChatModel

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
from openai_wrapper import num_tokens_from_string


def prompt_summarize_diff(diff_output):
    instructions = """
You are an expert programmer, write a descriptive and informative commit message following the Conventional Commits format for a recent code change, which is presented as the output of git diff --staged.

## Instructions
* Start with a commit summary following Conventional Commits format: type(scope): description
    * Type must be one of: feat, fix, docs, style, refactor, perf, test, build, ci, chore
    * Scope is optional and should be the main component being changed
    * Description should be concise but informative, using imperative mood
* Then add details in the body, separated by a blank line
* If you see bugs, start at the top of the file with
* When listing changes,
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
* If you see any of the following, skip them, or at most list them last as a single line
    * Changes to formatting/whitespace
    * Changes to imports
    * Changes to comments

Example:
feat(auth): add OAuth2 authentication flow

### BUGS: (Only include if bugs are seen)
    * List bugs
### Reason for change
    * reason
### Details
    * details

BREAKING CHANGE: (include only for breaking changes)
    * List breaking changes
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_output),
        ]
    )


async def a_build_commit():
    llms = langchain_helper.get_models(openai=True, claude=True)

    user_text = "".join(sys.stdin.readlines())
    tokens = num_tokens_from_string(user_text)

    if tokens < 8000:
        llms += [langchain_helper.get_model(llama=True)]

    if tokens < 4000:
        llms += [langchain_helper.get_model(llama=True)]

    def describe_diff(llm: BaseChatModel):
        return prompt_summarize_diff(user_text) | llm

    describe_diffs = await langchain_helper.async_run_on_llms(describe_diff, llms)

    for description, llm, duration in describe_diffs:
        print(
            f"# -- model: {langchain_helper.get_model_name(llm)} | {duration.total_seconds():.2f} seconds --"
        )
        print(description.content)


console = Console()
app = typer.Typer(no_args_is_help=True)


@logger.catch()
def app_wrap_loguru():
    app()


@app.command()
def build_commit(
    trace: bool = False,
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_build_commit())
    )


if __name__ == "__main__":
    app_wrap_loguru()
