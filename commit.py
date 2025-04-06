#!python3


import sys
import asyncio
from typing import List, Optional

from langchain_core import messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
from openai_wrapper import num_tokens_from_string

# Disable Rich's line wrapping
console = Console(width=10000)


def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped in the diff processing."""
    skip_files = ["cursor-logs", "chop-logs", "back-links.json"]
    return any(skip_file in file_path for skip_file in skip_files)


def filter_diff_content(diff_output: str) -> str:
    """Filter out diffs from files that should be skipped."""
    lines = diff_output.splitlines()
    filtered_lines = []
    skip_current_file = False

    for line in lines:
        if line.startswith("diff --git"):
            # Extract file path from diff header
            file_path = line.split(" b/")[-1]
            skip_current_file = should_skip_file(file_path)

        if not skip_current_file:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


# Define the structure for commit messages
class CommitMessage(BaseModel):
    summary: str = Field(
        ..., description="The commit summary line following Conventional Commits format"
    )
    bugs: Optional[List[str]] = Field(
        None, description="List of bugs identified in the code"
    )
    breaking_changes: Optional[List[str]] = Field(
        None, description="List of breaking changes"
    )
    cursor_rule_changes: Optional[List[str]] = Field(
        None, description="Changes to Cursor rules"
    )
    reasons: Optional[List[str]] = Field(None, description="Reasons for the changes")
    details: Optional[List[str]] = Field(None, description="Details about the changes")


def prompt_summarize_diff(diff_output, oneline=False):
    if oneline:
        instructions = """
You are an expert programmer, write a single-line commit message following the Conventional Commits format for a recent code change, which is presented as the output of git diff --staged.

The commit message should follow this format exactly:
type(scope): description

Where:
* Type must be one of: feat, fix, docs, style, refactor, perf, test, build, ci, chore
* Scope is optional and should be the main component being changed
* Description should be concise but informative, using imperative mood

Do not include any additional details, line breaks, or explanations. Just the single line.
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                messages.SystemMessage(content=instructions),
                messages.HumanMessage(content=diff_output),
            ]
        )
        return prompt | StrOutputParser()
    else:
        instructions = """
You are an expert programmer, write a descriptive and informative commit message following the Conventional Commits format for a recent code change, which is presented as the output of git diff --staged.

## Instructions
* Start with a commit summary following Conventional Commits format: type(scope): description
    * Type must be one of: feat, fix, docs, style, refactor, perf, test, build, ci, chore
    * Scope is optional and should be the main component being changed, skip if only one file changed
    * Description should be concise but informative, using imperative mood
* Then add details in the body, separated by a blank line
* If you see bugs, start at the top of the file with
* When listing changes,
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
* If any changes involve Cursor rules (files in .cursor/rules/ or with .mdc extension), include a special section for cursor_rule_changes
* If you see any of the following, skip them, or at most list them last as a single line
    * Changes to formatting/whitespace
    * Changes to imports
    * Changes to comments
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                messages.SystemMessage(content=instructions),
                messages.HumanMessage(content=diff_output),
            ]
        )
        return prompt


async def a_build_commit(oneline: bool = False, fast: bool = False):
    user_text = "".join(sys.stdin.readlines())
    # Filter out diffs from files that should be skipped
    filtered_text = filter_diff_content(user_text)

    if fast:
        # Use R1 (deepseek) only once for fast mode
        llms = [langchain_helper.get_model(llama=True)]
    elif oneline:
        # For oneline, just use Llama
        llms = [langchain_helper.get_model(llama=True)]
    else:
        llms = langchain_helper.get_models(
            openai=True, google=True, o3_mini=True, claude=True, structured=True
        )
        tokens = num_tokens_from_string(filtered_text)
        if tokens < 32_000:
            llms += [langchain_helper.get_model(llama=True)]

    def describe_diff(llm: BaseChatModel):
        if oneline:
            chain = prompt_summarize_diff(filtered_text, oneline)
            return chain | llm
        else:
            # Use the standard with_structured_output method for all models
            structured_llm = llm.with_structured_output(CommitMessage, include_raw=True)
            chain = prompt_summarize_diff(filtered_text, oneline) | structured_llm
            return chain

    describe_diffs = await langchain_helper.async_run_on_llms(describe_diff, llms)

    for result, llm, duration in describe_diffs:
        if oneline:
            # For oneline, result is just a string
            print(result)
        else:
            # For structured output, result contains raw, parsed, and parsing_error
            model_name = langchain_helper.get_model_name(llm)

            if result.get("parsing_error"):
                # Handle parsing error
                print(f"[bold red]Error parsing output from {model_name}:[/bold red]")
                print(f"[red]{result['parsing_error']}[/red]")
                print("\nRaw output:")
                print(
                    result["raw"].content
                    if hasattr(result["raw"], "content") and result["raw"].content
                    else result["raw"]
                )
                print("\n---\n")
                continue

            # Successfully parsed result
            commit = result["parsed"]
            print(commit.summary)

            if commit.bugs:
                print("\n**BUGS:**")
                for bug in commit.bugs:
                    print(f"* {bug}")

            if commit.breaking_changes:
                print("\n**BREAKING CHANGE:**")
                for change in commit.breaking_changes:
                    print(f"* {change}")

            if commit.cursor_rule_changes:
                print("\n**Cursor Rules Changes:**")
                for change in commit.cursor_rule_changes:
                    print(f"* {change}")

            if commit.reasons:
                print("\n**Reason for change**")
                for reason in commit.reasons:
                    print(f"* {reason}")

            if commit.details:
                print("\n**Details**")
                for detail in commit.details:
                    print(f"* {detail}")

        print(
            f"\ncommit message generated by {langchain_helper.get_model_name(llm)} in {duration.total_seconds():.2f} seconds"
        )


console = Console()
app = typer.Typer(no_args_is_help=True)


@logger.catch()
def app_wrap_loguru():
    app()


@app.command()
def build_commit(
    trace: bool = False,
    oneline: bool = typer.Option(
        False,
        "--oneline",
        help="Generate a single-line commit message using Llama only",
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Use Llama only once for faster processing"
    ),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_build_commit(oneline, fast))
    )


if __name__ == "__main__":
    app_wrap_loguru()
