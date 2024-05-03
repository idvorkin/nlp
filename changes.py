#!python3


import asyncio
import subprocess

from langchain_core import messages
from typing import Tuple

import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from pathlib import Path
import langchain_helper

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


openai_wrapper.setup_secret()


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
    if file.endswith("pdf"):
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


async def get_file_diff(file, first_commit_hash, last_commit_hash) -> Tuple[str, str]:
    """
    Asynchronously get the diff for a file, including the begin and end revision,
    and perform string parsing in Python to avoid using shell-specific commands.
    """
    if not Path(file).exists():
        ic(f"File {file} does not exist or has been deleted.")
        return file, ""

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


def tomorrow():
    # Get today's date
    today = datetime.now()
    # Calculate tomorrow's date by adding one day to today
    tomorrow_date = today + timedelta(days=1)
    # Format the date as a string in the format YYYY-MM-DD
    return tomorrow_date.strftime("%Y-%m-%d")


@app.command()
def changes(
    before=tomorrow(),
    after="7 days ago",
    trace: bool = False,
    gist: bool = False,
    openai: bool = False,
    google: bool = False,
    claude: bool = True,
):
    llm = langchain_helper.get_model(openai=openai, google=google, claude=claude)
    achanges_params = llm, before, after, gist
    if not trace:
        asyncio.run(achanges(*achanges_params))
        return

    from langchain_core.tracers.context import tracing_v2_enabled
    from langchain.callbacks.tracers.langchain import wait_for_all_tracers

    trace_name = openai_wrapper.tracer_project_name()
    with tracing_v2_enabled(project_name=trace_name) as tracer:
        ic("Using Langsmith:", trace_name)
        asyncio.run(achanges(*achanges_params))  # don't forget the second run
        ic(tracer.get_run_url())
    wait_for_all_tracers()


async def first_last_commit(before: str, after: str) -> Tuple[str, str]:
    git_log_command = f"git log --after='{after}' --before='{before}' --pretty='%H'"
    ic(git_log_command)

    # Execute the git log command
    process = await asyncio.create_subprocess_shell(
        git_log_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    git_output = stdout.decode().strip().split("\n")

    if not git_output:
        print("No commits found for the specified date range.")
        return ("", "")

    # Extract the first and last commit hashes
    first_commit = git_output[-1]
    last_commit = git_output[0]

    # Get the diff before the last commit
    git_cli_diff_before = f"git log {first_commit}^ -1 --pretty='%H'"
    # call it from a simple shell command
    first_diff = await asyncio.create_subprocess_shell(
        git_cli_diff_before,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await first_diff.communicate()
    first_commit = stdout.decode().strip()

    ic(first_commit, last_commit)
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


def prompt_report_from_diff_summary(diff_summary):
    instructions = """You are given a diff summary including what files changed, you will re-write it to be in the order of maximum importance

## Instructions

Begin by creating a a clickable markdown table of contents for all files. As an example

- [The first file](#the-first-file)
- [The second file](#the-second-file)

After the table of contents

* Maintain the input file format, but just change the order of files and the lines within files.
* The second line with 3 numbers indicates the diff addition, removal and file changes.
* Ensure all files listed in the input diff summary are still listed

Really minor changes should **not be listed**. Minor changes include.
* Changes to imports
* Exclude changes to spelling, grammar or punctuation in the summary
* Changes to wording, for example, exclude Changed "inprogress" to "in progress"

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_summary),
        ]
    )


# Function to create the prompt
def prompt_summarize_diff(file, diff_content, repo_path, end_rev):
    instructions = f""" You are an expert programmer.

    Summarize the changes for: {file}, permalink:{repo_path}/blob/{end_rev}/{file}

## Instructions

Have the first line be #### Filename on a single line
Have second line be file link, lines_added, lines_removed, ~lines change (but exclude changes in comments) on a single line
* When listing  changes,
    * Include the reason for the change if you can figure it out
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
* If you see any of the following, skip them, or at most list them last as a single line
    * Changes to formatting/whitespace
    * Changes to imports
    * Changes to comments
* If it's a new file, try to guess the purpose of the file
* Don't be tenative in your language

E.g. for the file _d/foo.md, with 5 lines added, 3 lines removed, and 34 lines changed (excluding changes to comments)

#### _d/foo.md

* [_d/foo.md](https://github.com/idvorkin/idvorkin.github.io/blob/3e8ee0cf75f9455c4f5da38d6bf36b221daca8cc/foo.md): +5, -3, ~34
* xyz changed from a to b
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_content),
        ]
    )


def create_markdown_table_of_contents(markdown_headers):
    def link_to_markdown(markdown_header):
        # '-' to "-", remove / and . from the link
        return (
            markdown_header.lower().replace(" ", "-").replace("/", "").replace(".", "")
        )

    return "\n".join(
        [f"- [{link}](#{link_to_markdown(link)})" for link in markdown_headers]
    )


async def achanges(llm: BaseChatModel, before, after, gist):
    # time the task
    start = datetime.now()
    repo_url, repo_name = get_repo_path()

    first, last = await first_last_commit(before, after)
    changed_files = await get_changed_files(first, last)

    file_diffs = await asyncio.gather(
        *[
            get_file_diff(file, first, last)
            for file in changed_files
            if not is_skip_file(file)
        ]
    )
    # add some rate limiting
    max_parallel = asyncio.Semaphore(100)

    async def concurrent_llm_call(file, diff_content):
        async with max_parallel:
            ic(f"running on {file}")
            return await (
                prompt_summarize_diff(
                    file, diff_content, repo_path=repo_url, end_rev=last
                )
                | llm
            ).ainvoke({})

    ai_invoke_tasks = [
        concurrent_llm_call(file, diff_content) for file, diff_content in file_diffs
    ]

    results = [result.content for result in await asyncio.gather(*ai_invoke_tasks)]

    # I think this can be done by the reorder_diff_summary command
    results.sort(key=lambda x: len(x), reverse=True)

    unranked_diff_report = "\n\n___\n\n".join(results)

    ic(unranked_diff_report)

    # diff_report = (
    #     (prompt_report_from_diff_summary(unranked_diff_report) | g_model)
    #     .invoke({})
    #     .content
    # )
    # print(diff_report)

    ## Pre-ranked output
    output = ""
    github_repo_diff_link = f"[{repo_name}]({repo_url}/compare/{first}...{last})"
    output = f"""
### Changes to {github_repo_diff_link} From [{after}] To [{before}]
* Model: {langchain_helper.get_model_name(llm)}
* Duration: {int((datetime.now() - start).total_seconds())} seconds
* Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S") }
___
{create_markdown_table_of_contents(changed_files)}
___
{unranked_diff_report}
    """
    print(output)

    output_file_path = Path(f"summary_{repo_name.split('/')[-1]}.md")
    with output_file_path.open("w", encoding="utf-8"):
        output_file_path.write_text(output)

    if gist:
        langchain_helper.to_gist(output_file_path)


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
