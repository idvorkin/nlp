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
    if file == "backlinks.json":
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


async def get_file_diff(file, revision_spec):
    """
    Asynchronously get the diff for a file, including the begin and end revision,
    and perform string parsing in Python to avoid using shell-specific commands.
    """
    if not Path(file).exists():
        ic(f"File {file} does not exist or has been deleted.")
        return file, "", "", ""
    ic(" ".join(["git", "log", "--reverse", "--format=%H", revision_spec, "--", file]))

    # Retrieve the entire commit history affecting the file within the revision_spec
    commit_process = await asyncio.create_subprocess_exec(
        "git",
        "log",
        "--reverse",
        "--format=%H",
        revision_spec,
        "--",
        file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_commits, _ = await commit_process.communicate()
    commit_hashes = stdout_commits.decode().strip().split("\n")
    ic(commit_hashes)

    # Parse the commit hashes to find the first (oldest) and last (most recent)
    first_commit_hash = commit_hashes[0] if commit_hashes else ""
    last_commit_hash = commit_hashes[-1] if commit_hashes else ""

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

    ic(file, stdout_diff.decode(), first_commit_hash, last_commit_hash)
    return file, stdout_diff.decode(), first_commit_hash, last_commit_hash


@app.command()
def changes(revision_spec="HEAD@{7 days ago}"):
    asyncio.run(achanges(revision_spec=revision_spec))


async def achanges(revision_spec):
    """
    Summarize the changes to all files in a given git revision specification.

    Args:
      revision_spec (str): The git revision specification to summarize changes for.

    This function will:
    - List out the changes to all files in the given revision specification.
    - Use the diff content to concisely explain the changes to each file.
    - Assume the function call_llm(prompt) exists for processing the diff summaries.
    """
    # First, we need to get a list of changed files for the given revision spec.
    model = ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name)

    _, base_path = get_repo_path()

    print(f"# Changes in {base_path} from {revision_spec}")

    changed_files_command = ["git", "diff", "--name-only", revision_spec]
    ic(changed_files_command)
    result = subprocess.run(changed_files_command, capture_output=True, text=True)
    changed_files = result.stdout.split("\n")

    # Iterate through the list of changed files and generate summaries.

    file_diffs = await asyncio.gather(
        *[
            get_file_diff(file, revision_spec)
            for file in changed_files
            if not is_skip_file(file)
        ]
    )
    ic(file_diffs[1])

    # sort by length to do biggest changes first
    file_diffs.sort(key=lambda x: len(x[1]), reverse=True)
    for file, diff_content, first_diff, last_diff in file_diffs:
        ic(file)

        prompt1 = f"""Summarize the changes for {file}

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
        result = (
            ChatPromptTemplate.from_messages([HumanMessage(content=prompt1)]) | model
        ).invoke({})
        print(result.content)

        # ic (diff_content)
        # Call the language model (or another service) to get a summary of the changes.
        # summary = call_llm(prompt)

        # Output the file name and its summary.
        # print(f"File: {file}\nSummary:\n{summary}\n")


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
