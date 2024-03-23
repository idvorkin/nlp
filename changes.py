#!python3


import os
import asyncio
import subprocess

from langchain_core.messages.human import HumanMessage

import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain_openai.chat_models import ChatOpenAI
import re
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


@app.command()
def changes(before="", after="7 days ago"):
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
    return ChatPromptTemplate.from_messages([HumanMessage(content=text)])


def diff_size(diff_summary):
    # Split the input into lines
    lines = diff_summary.splitlines()
    # Iterate through each line
    for line in lines:
        # Use a regular expression to find a line with exactly three numbers
        # The pattern is looking for optional spaces, optional non-digit characters (excluding the minus sign),
        # followed by an optional minus sign and one or more digits. This is repeated three times for the three numbers.
        line = line.replace("+", " ")
        line = line.replace("-", " ")
        line = line.replace("*", " ")
        line = line.replace(":", " ")
        line = line.replace(",", " ")

        pattern = r"^\s*(\d+)\s+(\d+)\s+(\d+).*$"
        match = re.search(pattern, line)
        if match:
            ic(match.groups())
            # If a match is found, sum the numbers
            # match.groups() contains all capturing groups of the match, which are the three numbers we're interested in
            total_sum = sum(int(number) for number in match.groups())
            return total_sum
    # If no matching line is found, return 0 or an appropriate value indicating no valid line was found
    return 0


async def achanges(before, after):
    # First, we need to get a list of changed files for the given revision spec.
    model = ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name)

    repo, base_path = get_repo_path()

    print(f"# Changes in {base_path} after [{after}] but before [{before}]")

    first, last = await first_last_commit(before, after)
    changed_files = await get_changed_files(first, last)
    print(f"[Changes in {base_path}]({repo}/compare/{first}...{last})")

    # Iterate through the list of changed files and generate summaries.

    file_diffs = await asyncio.gather(
        *[
            get_file_diff(file, first, last)
            for file in changed_files
            if not is_skip_file(file)
        ]
    )

    # sort by length to do biggest changes first
    file_diffs.sort(key=lambda x: len(x[1]), reverse=True)
    ai_invoke_tasks = [
        (diff_summary_prompt(file, diff_content) | model).ainvoke({})
        for file, diff_content in file_diffs
    ]

    results = await asyncio.gather(*ai_invoke_tasks)
    results.sort(key=lambda x: diff_size(x.content), reverse=True)
    for result in results:
        print(result.content)

        # ic (diff_content)
        # Call the language model (or another service) to get a summary of the changes.
        # summary = call_llm(prompt)

        # Output the file name and its summary.
        # print(f"File: {file}\nSummary:\n{summary}\n")


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
