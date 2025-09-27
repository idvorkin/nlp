#!python3


import sys
import asyncio

from langchain_core import messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

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


def clean_commit_message(message: str) -> str:
    """Remove any backticks or code block formatting from the commit message."""
    # Remove code block markers
    message = message.replace("```", "")

    # Remove any language specifiers that might appear after backticks
    lines = message.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip lines that only contain a language specifier like "git" or "bash"
        if line.strip() in ["git", "bash", "shell", "commit"]:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


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
Do not use backticks (```) or code formatting in your response.

Your response should be plain text only, it will be sent to git -m COMMIT_MSG directly
"""
    else:
        instructions = """
        # 🟢 Git Commit Message Generator Prompt (Enhanced)

You are an expert software engineer. Write a precise, descriptive, and informative commit message for a staged code change, using the **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)** format. The input is the output of `git diff --staged`.

---

## 📌 Rules

### 1. Commit Summary

Format:
`type(scope): description`

- `type` must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`
- `scope` is optional — use it if more than one file changed or if a clear module/component is being modified
- Use **imperative mood** ("Fix bug", not "Fixed bug")
- Be specific: avoid generic summaries like "fix bug"

---

### 2. Commit Body Structure

Follow this format and order *if applicable*:

```
## The Problem

Briefly explain the issue or unexpected behavior. Clarify assumptions or mismatches that caused the bug or need for change.

## The Fix

Explain what you changed and why it works. Describe important implementation decisions and tradeoffs.

## Testing

List how the change was tested or validated. Use bullet points or short narrative.

## Cursor Rules Changes:

If any `.cursor/rules/` or `.mdc` files changed, list and explain those here.

## BREAKING CHANGE:

Clearly state if this change breaks existing behavior and what consumers must do to adapt.

## Reason for change:

(Optional) Use this section for context or motivation not already captured above.
```

---

### 3. Content Guidelines

- Put the most important changes at the top
- Use `*` for bullet points
- Prefer clarity and conciseness over verbosity, but don’t skip meaningful rationale
- **Do not** mention discuss changes to the table of content.
- **Do not** mention discuss whitespace only edits
- **Do not** mention Pure comment or import order changes (unless essential — then list them last)
- **Do not** use backticks or code blocks — plain text only
- Only respond with the the commit - it  will be sent to git -m COMMIT_MSG directly

"""

    prompt = ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_output),
        ]
    )
    return prompt


async def a_build_commit(
    oneline: bool = False, fast: bool = False, kimi: bool = True, gpt_oss: bool = True, grok4_fast: bool = True
):
    user_text = "".join(sys.stdin.readlines())
    # Filter out diffs from files that should be skipped
    filtered_text = filter_diff_content(user_text)

    if fast:
        # Include Grok-4-Fast in fast mode for optimized results
        llms = [
            langchain_helper.get_model(llama=True),
            langchain_helper.get_model(gpt_oss=True),
            langchain_helper.get_model(kimi=True),
            langchain_helper.get_model(grok4_fast=True),
        ]
    elif oneline:
        # For oneline, optionally use grok4-fast if specified
        if grok4_fast:
            llms = [langchain_helper.get_model(grok4_fast=True)]
        else:
            llms = [langchain_helper.get_model(llama=True)]
    else:
        llms = langchain_helper.get_models(
            openai=True,
            google=True,
            claude=True,
            kimi=kimi,
            gpt_oss=gpt_oss,
            grok4_fast=grok4_fast,
        )
        tokens = num_tokens_from_string(filtered_text)
        if tokens < 32_000:
            llms += [langchain_helper.get_model(llama=True)]

    def describe_diff(llm: BaseChatModel):
        try:
            chain = (
                prompt_summarize_diff(filtered_text, oneline) | llm | StrOutputParser()
            )
            return chain
        except Exception as e:
            logger.error(
                f"Error with model {langchain_helper.get_model_name(llm)}: {e}"
            )
            return {"error": str(e)}

    describe_diffs = await langchain_helper.async_run_on_llms(describe_diff, llms)

    # Sort by duration (slowest first)
    describe_diffs_sorted = sorted(describe_diffs, key=lambda x: x[2], reverse=True)

    successful_results = 0

    for result, llm, duration in describe_diffs_sorted:
        model_name = langchain_helper.get_model_name(llm)

        try:
            if isinstance(result, dict) and "error" in result:
                # Handle case where the model function itself failed
                print(f"[bold red]Error with {model_name}:[/bold red]")
                print(f"[red]{result['error']}[/red]")
                print("\n---\n")
                continue

            # Result is now just a string
            cleaned_result = clean_commit_message(result)
            print(cleaned_result)
            successful_results += 1

            print(
                f"\ncommit message generated by {model_name} in {duration.total_seconds():.2f} seconds"
            )
            print("\n---\n")
        except Exception as e:
            # Catch any unexpected errors when processing results
            print(f"[bold red]Error processing result from {model_name}:[/bold red]")
            print(f"[red]{str(e)}[/red]")
            print("\n---\n")

    if successful_results == 0:
        print("[bold red]All models failed to generate commit messages.[/bold red]")
        return 1

    return 0


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
        False, "--fast", help="Use Llama 4 and GPT-OSS for faster processing"
    ),
    kimi: bool = typer.Option(
        True, "--kimi/--no-kimi", help="Use Kimi model (default: enabled)"
    ),
    gpt_oss: bool = typer.Option(
        True, "--gpt-oss/--no-gpt-oss", help="Use GPT-OSS-120B model (default: enabled)"
    ),
    grok4_fast: bool = typer.Option(
        True, "--grok4-fast/--no-grok4-fast",
        help="Use XAI Grok-4-Fast model (default: enabled)"
    ),
):
    def run_build():
        result = asyncio.run(a_build_commit(oneline, fast, kimi, gpt_oss, grok4_fast))
        if result != 0:
            raise typer.Exit(code=result)

    langchain_helper.langsmith_trace_if_requested(trace, run_build)


if __name__ == "__main__":
    app_wrap_loguru()
