#!python3


import os
import asyncio
import subprocess
from contextlib import contextmanager

from langchain_core import messages

import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from pathlib import Path
import json

console = Console()
app = typer.Typer()


class Diff(BaseModel):
    FilePath: Path
    StartRevision: str
    EndRevision: str
    DiffContents: str


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


def setup_secret():
    secret_file = Path.home() / "gits/igor2/secretBox.json"
    SECRETS = json.loads(secret_file.read_text())
    os.environ["OPENAI_API_KEY"] = SECRETS["openai"]


setup_secret()


@logger.catch()
def app_wrap_loguru():
    app()


def is_skip_file(file):
    if file.strip() == "":
        return True
    file_path = Path(file)
    # Verify the file exists before proceeding.
    if not file_path.exists():
        ic(f"File {file} does not exist or has been deleted.")
        return True

    if file.endswith("ipynb"):
        ic("ipynb not supported yet")
        return True
    if file == "back-links.json":
        return True

    return False


def get_repo_path():
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )

    # Assuming the URL is in the form: https://github.com/idvorkin/bob or git@github.com:idvorkin/bob
    repo_url = result.stdout.strip()
    base_path = "Unknown"
    if repo_url.startswith("https"):
        base_path = repo_url.split("/")[-2] + "/" + repo_url.split("/")[-1]
    elif repo_url.startswith("git@"):
        base_path = repo_url.split(":")[1]
        base_path = base_path.replace(".git", "")
    return repo_url, base_path


async def get_file_diff(file, first_commit_hash, last_commit_hash):
    """
    Asynchronously get the diff for a file, including the begin and end revision,
    and perform string parsing in Python to avoid using shell-specific commands.
    """
    if not Path(file).exists():
        ic(f"File {file} does not exist or has been deleted.")
        return file, "", "", ""

    # Now retrieve the diff using the first and last commit hashes
    diff_process = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        first_commit_hash,
        last_commit_hash,
        "--",
        file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_diff, _ = await diff_process.communicate()

    return file, stdout_diff.decode()


@contextmanager
def noop_context():
    yield


@app.command()
def changes(before="", after="7 days ago", trace: bool = False):
    tracer_function = noop_context
    if trace:
        ic("using wandb", trace)
        from langchain_community.callbacks import wandb_tracing_enabled

        tracer_function = wandb_tracing_enabled

    with tracer_function():
        asyncio.run(achanges(before, after))


async def first_last_commit(before, after):
    git_log_command = f"git log --since='{after}' --until='{before}' --pretty='%H'"

    # Execute the git log command
    process = await asyncio.create_subprocess_shell(
        git_log_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    git_output = stdout.decode().strip().split("\n")

    if not git_output or len(git_output) < 2:
        print("Insufficient commits found for the specified date range.")
        return

    # Extract the first and last commit hashes
    first_commit = git_output[-1]
    last_commit = git_output[0]
    return first_commit, last_commit


async def get_changed_files(first_commit, last_commit):
    git_diff_command = f"git diff --name-only {first_commit} {last_commit}"

    # Execute the git diff command
    process = await asyncio.create_subprocess_shell(
        git_diff_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    changed_files_output = stdout.decode().strip()

    # Split the output into a list of file paths
    changed_files = changed_files_output.split("\n") if changed_files_output else []

    return changed_files


# Function to create the prompt


def reorder_diff_summary(diff_summary):
    text = """You are given a diff summary including what files changed, you will re-write it to be in the order of maximum importance

## Instructions

Maintain the input file format, but just change the order of files and the lines within files.
The second line with 3 numbers indicates the diff addition, removal and file changes.
Really minor changes should **not be listed**. Minor changes include.
* Changes to imports
* Exclude changes to spelling, grammar or punctuation in the summary
* Changes to wording, for example, exclude Changed "inprogress" to "in progress"
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=text),
            messages.HumanMessage(content=diff_summary),
        ]
    )


# Function to create the prompt
def diff_summary_prompt(file, diff_content):
    text = f"""Summarize the changes for {file}

## Instructions

Have the first line be ### Filename on a single line
Have second line be lines_added, lines_removed, lines change (but exclude changes in comments) on a single line
For the remaining lines use a markdown list
When having larger changes add details by including sub bullets.
List the changes in the list in order of impact. The most impactful/major changes should go first, and minor changes should go last.
Really minor changes should **not be listed**. Minor changes include.
* Changes to imports
* Exclude changes to spelling, grammar or punctuation in the summary
* Changes to wording, for example, exclude Changed "inprogress" to "in progress"

E.g. for the file foo.md

### foo.md
+ 5, -3, * 34:
- xyz changed from a to b


## Diff Contents
{diff_content}"""
    return ChatPromptTemplate.from_messages([messages.HumanMessage(content=text)])


async def achanges(before, after):
    model = ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name)

    repo, base_path = get_repo_path()
    print(f"# Changes in {base_path} after [{after}] but before [{before}]")

    first, last = await first_last_commit(before, after)
    changed_files = await get_changed_files(first, last)

    print(f"[Changes in {base_path}]({repo}/compare/{first}...{last})")

    print("## Pass 1 Diff Summary")

    file_diffs = await asyncio.gather(
        *[
            get_file_diff(file, first, last)
            for file in changed_files
            if not is_skip_file(file)
        ]
    )
    ai_invoke_tasks = [
        (diff_summary_prompt(file, diff_content) | model).ainvoke({})
        for file, diff_content in file_diffs
    ]

    results = await asyncio.gather(*ai_invoke_tasks)

    # I think this can be done by the reorder_diff_summary command
    # results.sort(key=lambda x: diff_size(x.content), reverse=True)

    initial_diff_report = "\n".join([result.content for result in results])

    print(initial_diff_report)

    print("## Re-Ranked Diff Summary ")

    ranked_output = (
        (reorder_diff_summary(initial_diff_report) | model).invoke({}).content
    )
    print(ranked_output)

    # print ("## Operational Stats ")


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
