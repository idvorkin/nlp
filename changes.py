#!python3


import asyncio
import subprocess
import requests

from langchain_core import messages
from typing import List, Tuple

import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from github_helper import get_latest_github_commit_url, get_repo_info


from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from loguru import logger
from pydantic import BaseModel
from rich import print as rich_print
from rich.console import Console
from pathlib import Path
import langchain_helper
from contextlib import contextmanager
import os

console = Console()
app = typer.Typer(no_args_is_help=True)


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

    if file.startswith("assets/js/idv-blog-module"):
        ic("Skip generated module file")
        return True

    if file.endswith(".js.map"):
        ic("Skip mapping files")
        return True
    if file == "back-links.json":
        return True

    return False




async def get_file_diff(file, first_commit_hash, last_commit_hash) -> Tuple[str, str]:
    """
    Asynchronously get the diff for a file, including the begin and end revision,
    and perform string parsing in Python to avoid using shell-specific commands.
    Skip diffs larger than 100,000 characters.
    """
    if not Path(file).exists():
        ic(f"File {file} does not exist or has been deleted.")
        return file, ""

    # Use nbdiff for Jupyter notebooks to ignore outputs
    if file.endswith(".ipynb"):
        diff_process = await asyncio.create_subprocess_exec(
            "nbdiff",
            "--ignore-outputs",
            first_commit_hash,
            last_commit_hash,
            "--",
            file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
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

    diff_content = stdout_diff.decode()

    if len(diff_content) > 30_000:
        return (
            file,
            f"Diff skipped: size exceeds 30,000 characters (actual size: {len(diff_content)} characters)",
        )

    return file, diff_content


def tomorrow():
    # Get today's date
    today = datetime.now()
    # Calculate tomorrow's date by adding one day to today
    tomorrow_date = today + timedelta(days=1)
    # Format the date as a string in the format YYYY-MM-DD
    return tomorrow_date.strftime("%Y-%m-%d")


@app.command()
def changes(
    directory: Path = Path("."),
    before=tomorrow(),
    after="7 days ago",
    trace: bool = False,
    gist: bool = True,
    openai: bool = True,
    claude: bool = True,
    google: bool = True,
    llama: bool = False,
):
    llms = langchain_helper.get_models(openai=openai, claude=claude, google=google)
    
    # Add llama only if requested
    if llama:
        llms.append(langchain_helper.get_model(llama=True))
        
    achanges_params = llms, before, after, gist

    # check if direcotry is in a git repo, if so go to the root of the repo
    is_git_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True
    ).stdout.strip()
    if is_git_repo == b"true":
        ic("Inside a git repo, moving to the root of the repo")
        subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True)
        directory = Path(
            subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
            .stdout.strip()
            .decode()
        )
    else:
        ic("Not in a git repo, using the current directory")

    with DirectoryContext(directory):
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


def prompt_summarize_diff_summaries(diff_summary):
    instructions = """
<instructions>


Please summarize the passed in report file (the actual report will be appended after this output). The summary should include:

<file_link_instructions_and_example>

When making file links, keep the _'s, here are valid examples:C

- [.gitignore](#gitignore)
- [_d/ai-journal.md](#_dai-journalmd)
- [_d/mood.md](#_dmoodmd)
- [_d/time-off-2024-08.md](#_dtime-off-2024-08md)
- [_d/time_off.md](#_dtime_offmd)
- [_includes/scripts.html](#_includesscriptshtml)
- [_posts/2017-04-12-happy.md](#_posts2017-04-12-happymd)
- [_td/slow_zsh.md](#_tdslow_zshmd)
- [graph.html](#graphhtml)
- [justfile](#justfile)
- [package.json](#packagejson)
</file_link_instructions_and_example>

<understanding_passed_in_report>
* Contains the changed diffs
* A line like: graph.html: +4, -1, ~3, tells you the changes in the file. This means 4 lines added, 1 removed, and 3 changed. It gives a tip on how big the changes are.

</understanding_passed_in_report>


<summary_instructions>
A summary of the higher level changes/intent of changes across all the files (e.g. implemented features).
    * Markdown files except readmes (especially in _d, _posts, _td) should come before any code changes in the summary
    * It should be divided by logical changes, not physical files.
    * Changes refererring to files should have clickable link to the lower section.
    * It should be ordered by importance


When summarizing, if working on a cli tool and it gets new commands. Be sure to include those at the top.

</summary_instructions>

<summary_example>
### Summary

* Line 1 - ([file](#link-to-file-in-the-below-report), [file](#link-to-file-in-the-below-report))
* Line 2
</summary_example>

<table_of_content_instructions>
A table of changes with clickable links to each section.
Order files by magnitude/importance of change, use same rules as with summary
</table_of_content_instructions>

<table_of_content_example>
### Table of Changes (LLM)

* [file](#link-to-file-in-the-below-report)
    * Most important change #1
    * Most important change #2
    * Most important change #3 (if major)
    * Most important change #4 (if major)
</table_of_content_example>


1. Remember don't include the report below, it will be added afterwards

</instructions>
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_summary),
        ]
    )


# Function to create the prompt
def prompt_summarize_diff(file, diff_content, repo_path, end_rev):
    instructions = f""" You are an expert programmer, who is charged with explaining code changes concisely.

You are  summarizing the passed in changes for: {file}, permalink:{repo_path}/blob/{end_rev}/{file}

<instructions>

* Have the first line be #### Filename on a single line
* Have second line be file link, lines_added, lines_removed, ~lines change (but exclude changes in comments) on a single line
* Have third line be a TL;DR of the changes
* If a new file is added, The TL;DR should describe the reason for the file
* When listing  changes,
    * Include the reason for the change if you can figure it out
    * Put them in the order of importance
    * Use unnumbered lists as the user will want to reorder them
* Do not include minor changes such as in the report
    * Changes to formatting/whitespace
    * Changes to imports
    * Changes to comments
* Be assertive in your language
* Start with why. For example
    * Instead of: Removed the requests and html2text imports and the associated get_text function, consolidating text retrieval logic into langchain_helper.get_text_from_path_or_stdin. This simplifies the code and removes the dependency on external libraries.
    *  Use: Remove dependancy on external libraries by consolidating retrieval logic into  langchain_helper.get_text_from_path_or_stdin

    * Instead of: Changed the prompt formatting instructions to clarify that groups should be titled with their actual names instead of the word "group". This enhances clarity for the user.
    * Use: Enhance clarity by using actual group names instead of the word "group" in the prompt formatting instructions.
</instructions>

<example>
E.g. for the file _d/foo.md, with 5 lines added, 3 lines removed, and 34 lines changed (excluding changes to comments)

#### _d/foo.md

[_d/foo.md](https://github.com/idvorkin/idvorkin.github.io/blob/3e8ee0cf75f9455c4f5da38d6bf36b221daca8cc/foo.md): +5, -3, ~34

TLDR: blah blah blah

* Reason for change, chanage
    * Sub details of change
</example>



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


@contextmanager
def DirectoryContext(directory: Path):
    original_directory = Path.cwd()
    try:
        if directory != ".":
            directory = Path(directory)
            if not directory.exists():
                print(f"Directory {directory} does not exist.")
                return
            os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


async def achanges(llms: List[BaseChatModel], before, after, gist):
    ic("v 0.0.4")
    start = datetime.now()
    repo_info = get_repo_info()

    # Run first_last_commit and get_changed_files in parallel
    first_last, _ = await asyncio.gather(
        first_last_commit(before, after),
        get_changed_files("", "")  # Placeholder, will be updated
    )
    first, last = first_last
    
    # Get updated changed files with correct commit hashes
    changed_files = await get_changed_files(first, last)
    changed_files = [file for file in changed_files if not is_skip_file(file)]

    # Get all file diffs in parallel
    file_diffs = await asyncio.gather(
        *[get_file_diff(file, first, last) for file in changed_files]
    )

    # Process all models in parallel
    async def process_model(llm):
        model_start = datetime.now()
        max_parallel = asyncio.Semaphore(100)

        async def concurrent_llm_call(file, diff_content):
            async with max_parallel:
                ic(f"running on {file} with {langchain_helper.get_model_name(llm)}")
                return await (
                    prompt_summarize_diff(
                        file, diff_content, repo_path=repo_info.url, end_rev=last
                    )
                    | llm
                ).ainvoke({})

        # Run all file analyses for this model in parallel
        ai_invoke_tasks = [
            concurrent_llm_call(file, diff_content) for file, diff_content in file_diffs
        ]
        results = [result.content for result in await asyncio.gather(*ai_invoke_tasks)]
        results.sort(key=lambda x: len(x), reverse=True)
        code_based_diff_report = "\n\n___\n\n".join(results)
        
        # Get summary for this model
        summary_all_diffs = await (
            (prompt_summarize_diff_summaries(code_based_diff_report) | llm)
            .ainvoke({})
        )
        summary_content = summary_all_diffs.content if hasattr(summary_all_diffs, 'content') else summary_all_diffs
        
        model_end = datetime.now()
        return {
            "model": llm,
            "model_name": langchain_helper.get_model_name(llm),
            "analysis_duration": model_end - model_start,
            "diff_report": code_based_diff_report,
            "summary": summary_content
        }

    # Run all models in parallel
    analysis_results = await asyncio.gather(*(process_model(llm) for llm in llms))

    # Create all output files in parallel
    async def write_model_summary(result):
        model_name = result["model_name"]
        safe_model_name = model_name.lower().replace(".", "-")
        
        github_repo_diff_link = f"[{repo_info.name}]({repo_info.url}/compare/{first}...{last})"
        model_output = f"""
### Changes to {github_repo_diff_link} From [{after}] To [{before}]
* Model: {model_name}
* Duration: {int(result["analysis_duration"].total_seconds())} seconds
* Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
___
### Table of Contents (code)
{create_markdown_table_of_contents(changed_files)}
___
{result["summary"]}
___
{result["diff_report"]}
"""
        summary_path = Path(".") / f"summary_{safe_model_name}.md"
        await asyncio.to_thread(summary_path.write_text, model_output)
        return summary_path

    # Write all summary files in parallel
    summary_paths = await asyncio.gather(
        *(write_model_summary(result) for result in analysis_results)
    )

    # Create and write overview file
    overview_filename = f"a_{repo_info.name.split('/')[-1]}--overview"
    overview_path = Path(".") / f"{overview_filename}.md"
    
    today = datetime.now().strftime("%Y-%m-%d")
    github_repo_diff_link = f"[{repo_info.name}]({repo_info.url}/compare/{first}...{last})"
    overview_content = f"""*🔄 via [changes.py]({get_latest_github_commit_url(repo_info.name, "changes.py")}) - {today}*

Changes to {github_repo_diff_link} From [{after}] To [{before}]

| Model | Analysis Duration (seconds) | Output Size (KB) |
|-------|---------------------------|-----------------|
"""
    
    for result in sorted(analysis_results, key=lambda x: x["analysis_duration"].total_seconds(), reverse=True):
        model_name = result["model_name"]
        safe_name = model_name.lower().replace(".", "-")
        duration = int(result["analysis_duration"].total_seconds())
        output_size = len(result["diff_report"] + result["summary"]) / 1024  # Convert to KB
        overview_content += f"| [{model_name}](#file-summary_{safe_name}-md) | {duration} | {output_size:.1f} |\n"

    # Write overview file
    await asyncio.to_thread(overview_path.write_text, overview_content)

    files_to_gist = [overview_path] + list(summary_paths)

    if gist:
        await asyncio.to_thread(langchain_helper.to_gist_multiple, files_to_gist)
    else:
        print(overview_content)
        for result in analysis_results:
            model_name = result["model_name"]
            print(f"\n=== Analysis by {model_name} ===\n")
            print(result["diff_report"])


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
