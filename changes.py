#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "icecream",
#     "langchain",
#     "langchain-community",
#     "langchain-core",
#     "langchain-openai",
#     "langchain-google-genai",
#     "langchain-anthropic",
#     "langchain-groq",
#     "loguru",
#     "openai",
#     "pydantic",
#     "pudb",
#     "requests",
#     "rich",
#     "typer",
# ]
# ///


import asyncio
import subprocess
import tempfile
import math

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
from langchain_community.chat_models import ChatOpenAI

from loguru import logger
from pydantic import BaseModel
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


def is_skip_file(file, only_pattern=None, verbose=False):
    if file.strip() == "":
        return True

    # If only_pattern is specified, skip files that don't match the pattern
    if only_pattern:
        from fnmatch import fnmatch

        if not fnmatch(file, only_pattern):
            if verbose:
                ic(f"Skip {file} as it doesn't match pattern {only_pattern}")
            return True

    file_path = Path(file)
    # Verify the file exists before proceeding.
    if not file_path.exists():
        if verbose:
            ic(f"File {file} does not exist or has been deleted.")
        return True

    # Skip cursor-logs and chop-logs directories
    skip_paths = ["cursor-logs", "chop-logs", "alist", "back-links.json"]
    if any(skip_path in str(file_path) for skip_path in skip_paths):
        if verbose:
            ic(f"Skip logs file: {file}")
        return True

    # Skip binary and media files
    binary_extensions = {
        # Images
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".webp",
        # Audio
        ".mp3",
        ".wav",
        ".ogg",
        ".m4a",
        ".flac",
        # Video
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".mkv",
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        # Archives
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".bz2",
        # Executables
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        # Other binary
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".pyc",
    }

    if file_path.suffix.lower() in binary_extensions:
        if verbose:
            ic(f"Skip binary file: {file}")
        return True

    if file.startswith("assets/js/idv-blog-module"):
        if verbose:
            ic("Skip generated module file")
        return True

    if file.endswith(".js.map"):
        if verbose:
            ic("Skip mapping files")
        return True
    if file == "back-links.json":
        return True

    return False


async def get_file_diff(
    file, first_commit_hash, last_commit_hash, verbose=False
) -> Tuple[str, str]:
    """
    Asynchronously get the diff for a file, including the begin and end revision,
    and perform string parsing in Python to avoid using shell-specific commands.
    Skip diffs larger than 100,000 characters.
    """
    if verbose:
        ic(f"++ Starting diff for: {file}")
    if not Path(file).exists():
        if verbose:
            ic(f"File {file} does not exist or has been deleted.")
        return file, ""

    # First check if git considers it a binary file
    if verbose:
        ic(f"Checking if {file} is binary")
    is_binary_cmd = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "--numstat",
        first_commit_hash,
        last_commit_hash,
        "--",
        file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = await is_binary_cmd.communicate()
    # If the file is binary, git outputs "-" for both additions and deletions
    if "-\t-\t" in stdout.decode():
        if verbose:
            ic(f"Git reports {file} as binary, skipping diff")
        return file, f"Skipped binary file: {file}"

    # Get file size before and after to provide context
    if verbose:
        ic(f"Getting file sizes for {file}")
    file_size_before = 0
    file_size_after = Path(file).stat().st_size if Path(file).exists() else 0

    try:
        # Try to get size of file in previous commit
        if verbose:
            ic(f"Getting previous size for {file}")
        size_cmd = await asyncio.create_subprocess_exec(
            "git",
            "cat-file",
            "-s",
            f"{first_commit_hash}:{file}",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await size_cmd.communicate()
        file_size_before = int(stdout.decode().strip()) if stdout else 0
    except Exception as e:
        if verbose:
            ic(f"Error getting previous size for {file}: {str(e)}")
        pass

    # Use nbdiff for Jupyter notebooks to ignore outputs
    if verbose:
        ic(f"Starting diff process for {file}")

    async def run_with_timeout(cmd, timeout_seconds=30):
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            if verbose:
                ic(
                    f"Command timed out after {timeout_seconds} seconds: {' '.join(cmd)}"
                )
            return None, None

    if file.endswith(".ipynb"):
        if verbose:
            ic("Checking if nbdiff is available")
        # Check if nbdiff is available
        which_process = await asyncio.create_subprocess_exec(
            "which",
            "nbdiff",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await which_process.communicate()

        if stdout and stdout.decode().strip():
            if verbose:
                ic(f"Using nbdiff for {file}")
            stdout_diff, stderr = await run_with_timeout(
                [
                    "nbdiff",
                    "--ignore-outputs",
                    first_commit_hash,
                    last_commit_hash,
                    "--",
                    file,
                ]
            )

            if stdout_diff is None:  # timeout occurred
                if verbose:
                    ic(f"nbdiff timed out, falling back to git diff for {file}")
                use_git_diff = True
            else:
                use_git_diff = False
        else:
            if verbose:
                ic(f"nbdiff not found, falling back to git diff for {file}")
            use_git_diff = True
    else:
        use_git_diff = True

    if use_git_diff:
        if verbose:
            ic(f"Using git diff for {file}")
        stdout_diff, stderr = await run_with_timeout(
            [
                "git",
                "diff",
                first_commit_hash,
                last_commit_hash,
                "--",
                file,
            ]
        )

        if stdout_diff is None:  # timeout occurred
            if verbose:
                ic(f"Git diff timed out for {file}")
            return file, f"Diff timed out for {file}"

    if stderr and verbose:
        ic(f"Stderr from diff for {file}: {stderr.decode('utf-8', errors='replace')}")

    if not stdout_diff:
        if verbose:
            ic(f"No diff output for {file}")
        return file, f"No diff output for {file}"

    if verbose:
        ic(f"Decoding diff output for {file}")
    try:
        diff_content = stdout_diff.decode("utf-8")
    except UnicodeDecodeError:
        if verbose:
            ic(f"Failed to decode diff for {file}, likely binary")
        return file, f"Failed to decode diff for {file}, likely binary"

    if len(diff_content) > 100_000:
        size_before_kb = file_size_before / 1024
        size_after_kb = file_size_after / 1024
        return (
            file,
            f"Diff skipped: size exceeds 100,000 characters (actual size: {len(diff_content):,} characters)\n"
            f"File size changed from {size_before_kb:.1f}KB to {size_after_kb:.1f}KB",
        )

    if verbose:
        ic(f"-- Completed diff for: {file}")
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
    google: bool = False,
    google_flash: bool = True,
    google_think: bool = typer.Option(
        True, help="Google Gemini 2.5 Flash with thinking (8192 tokens - MEDIUM level)"
    ),
    google_think_low: bool = typer.Option(
        False, help="Google Gemini 2.5 Flash with low thinking (1024 tokens)"
    ),
    google_think_medium: bool = typer.Option(
        False, help="Google Gemini 2.5 Flash with medium thinking (8192 tokens)"
    ),
    google_think_high: bool = typer.Option(
        False, help="Google Gemini 2.5 Flash with high thinking (24576 tokens)"
    ),
    llama: bool = True,
    o4_mini: bool = True,
    fast: bool = typer.Option(False, help="Fast analysis using only Llama model"),
    only: str = None,
    verbose: bool = False,
    completion_ratio: float = typer.Option(
        0.5,
        help="Fraction of models that must finish before returning early",
    ),
):
    # If fast is True, override other model selections to use only llama
    if fast:
        openai = False
        claude = False
        google = False
        google_flash = False
        google_think = False
        google_think_low = False
        google_think_medium = False
        google_think_high = False
        llama = True
        o4_mini = False
        if verbose:
            print("Fast mode: using only Llama model for quick analysis")

    llms = langchain_helper.get_models(
        openai=openai,
        claude=claude,
        google=google,
        google_flash=google_flash,
        google_think=google_think,
        google_think_low=google_think_low,
        google_think_medium=google_think_medium,
        google_think_high=google_think_high,
        o4_mini=o4_mini,
        llama=llama,
    )

    # If no models are selected, provide a helpful error message
    if not llms:
        print("Error: No models selected. Please enable at least one model.")
        return

    achanges_params = llms, before, after, gist, only, verbose, completion_ratio

    # check if direcotry is in a git repo, if so go to the root of the repo
    is_git_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True
    ).stdout.strip()
    if is_git_repo == b"true":
        if verbose:
            ic("Inside a git repo, moving to the root of the repo")
        directory = Path(
            subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
            .stdout.strip()
            .decode()
        )
    else:
        if verbose:
            ic("Not in a git repo, using the current directory")

    with DirectoryContext(directory):
        if not trace:
            asyncio.run(achanges(*achanges_params))
            return

        from langchain_core.tracers.context import tracing_v2_enabled
        from langchain.callbacks.tracers.langchain import wait_for_all_tracers

        trace_name = openai_wrapper.tracer_project_name()
        with tracing_v2_enabled(project_name=trace_name) as tracer:
            if verbose:
                ic("Using Langsmith:", trace_name)
            asyncio.run(achanges(*achanges_params))
            if verbose:
                ic(tracer.get_run_url())
        wait_for_all_tracers()


async def first_last_commit(before: str, after: str, verbose=False) -> Tuple[str, str]:
    git_log_command = f"git log --after='{after}' --before='{before}' --pretty='%H'"
    if verbose:
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

    if verbose:
        ic(first_commit, last_commit)
    return first_commit, last_commit


async def get_changed_files(first_commit, last_commit, verbose=False):
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


@contextmanager
def TempDirectoryContext():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


def sanitize_model_name(model_name):
    """Create a safe filename/ID from a model name by replacing problematic characters."""
    return model_name.lower().replace(".", "-").replace("/", "-")


async def gather_until_half_complete(
    tasks, grace_seconds: int = 10, completion_ratio: float = 0.5
):
    """Gather results from tasks but return early.

    Parameters
    ----------
    tasks : Iterable[Awaitable]
        The tasks to await.
    grace_seconds : int, optional
        Extra seconds to wait after the threshold is reached, by default ``10``.
    completion_ratio : float, optional
        Ratio of tasks that must complete before returning, by default ``0.5``.

    Returns
    -------
    tuple[list[Any], int, int]
        A tuple containing the list of completed task results, the number of
        tasks completed and the total number of tasks.
    """

    task_objs = {asyncio.create_task(t) for t in tasks}
    completed: set[asyncio.Task] = set()
    total_tasks = len(task_objs)
    threshold = max(1, math.ceil(total_tasks * completion_ratio))

    # Wait until enough tasks finish
    while task_objs and len(completed) < threshold:
        done, task_objs = await asyncio.wait(task_objs, return_when=asyncio.FIRST_COMPLETED)
        completed.update(done)

    if task_objs:
        done, task_objs = await asyncio.wait(task_objs, timeout=grace_seconds)
        completed.update(done)

    for t in task_objs:
        t.cancel()
    await asyncio.gather(*task_objs, return_exceptions=True)

    results = []
    for t in completed:
        if not t.cancelled():
            try:
                results.append(t.result())
            except Exception as e:
                logger.error(f"Task error: {e}")

    return results, len(completed), total_tasks


async def achanges(
    llms: List[BaseChatModel],
    before,
    after,
    gist,
    only: str = None,
    verbose: bool = False,
    completion_ratio: float = 0.5,
):
    if verbose:
        ic("v 0.0.4")
    repo_info = get_repo_info(for_file_changes=True)
    repo_path = Path.cwd()

    # Add file operation semaphore
    file_semaphore = asyncio.Semaphore(50)  # Limit concurrent file operations
    if verbose:
        ic("Created file semaphore with limit of 50")

    # Run first_last_commit and get_changed_files in parallel
    first_last, _ = await asyncio.gather(
        first_last_commit(before, after, verbose),
        get_changed_files("", "", verbose),  # Placeholder, will be updated
    )
    first, last = first_last

    # Get updated changed files with correct commit hashes
    changed_files = await get_changed_files(first, last, verbose)
    changed_files = [
        file for file in changed_files if not is_skip_file(file, only, verbose)
    ]

    # Modify get_file_diff calls to use semaphore
    async def get_file_diff_with_semaphore(file, first, last):
        if verbose:
            ic(f"Waiting for file semaphore to process: {file}")
        async with file_semaphore:
            if verbose:
                ic(f"Acquired file semaphore for: {file}")
            result = await get_file_diff(file, first, last, verbose)
            if verbose:
                ic(f"Released file semaphore for: {file}")
            return result

    # Get all file diffs in parallel with semaphore
    file_diffs = await asyncio.gather(
        *[get_file_diff_with_semaphore(file, first, last) for file in changed_files]
    )

    # Process all models in parallel
    async def process_model(llm):
        model_start = datetime.now()
        max_parallel = asyncio.Semaphore(100)

        # Special handling for o4-mini model
        if isinstance(llm, ChatOpenAI) and llm.model_name == "o4-mini-2025-04-16":
            llm.model_kwargs = {}  # Clear any default parameters like temperature

        async def concurrent_llm_call(file, diff_content):
            model_name = langchain_helper.get_model_name(llm)
            if verbose:
                ic(f"Waiting for max_parallel semaphore for {file} with {model_name}")
            async with max_parallel:
                if verbose:
                    ic(f"Acquired max_parallel semaphore for {file} with {model_name}")
                    ic(f"++ LLM call start: {file} with {model_name}")
                result = await (
                    prompt_summarize_diff(
                        file, diff_content, repo_path=repo_path, end_rev=last
                    )
                    | llm
                ).ainvoke({})
                if verbose:
                    ic(f"-- LLM call end: {file} with {model_name}")
                    ic(f"Released max_parallel semaphore for {file} with {model_name}")
                return result

        # Run all file analyses for this model in parallel
        ai_invoke_tasks = [
            concurrent_llm_call(file, diff_content) for file, diff_content in file_diffs
        ]
        results = [result.content for result in await asyncio.gather(*ai_invoke_tasks)]
        results.sort(key=lambda x: len(x), reverse=True)
        code_based_diff_report = "\n\n___\n\n".join(results)

        model_name = langchain_helper.get_model_name(llm)
        # Get summary for this model
        if verbose:
            ic(f"++ LLM summary call start with {model_name}")
        summary_all_diffs = await (
            prompt_summarize_diff_summaries(code_based_diff_report) | llm
        ).ainvoke({})
        if verbose:
            ic(f"-- LLM summary call end with {model_name}")
        summary_content = (
            summary_all_diffs.content
            if hasattr(summary_all_diffs, "content")
            else summary_all_diffs
        )

        model_end = datetime.now()
        return {
            "model": llm,
            "model_name": model_name,
            "analysis_duration": model_end - model_start,
            "diff_report": code_based_diff_report,
            "summary": summary_content,
        }

    # Run all models in parallel but return early based on completion_ratio
    grace_seconds = 10
    analysis_results, completed_models, total_models = await gather_until_half_complete(
        [process_model(llm) for llm in llms],
        grace_seconds=grace_seconds,
        completion_ratio=completion_ratio,
    )

    # Create all output files in parallel
    async def write_model_summary(result, temp_dir: Path):
        model_name = result["model_name"]
        safe_model_name = sanitize_model_name(model_name)
        summary_path = temp_dir / f"z_{safe_model_name}.md"

        github_repo_diff_link = (
            f"[{repo_info.name}]({repo_info.url}/compare/{first}...{last})"
        )
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
        # Ensure parent directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(summary_path.write_text, model_output)
        return summary_path

    with TempDirectoryContext() as temp_dir:
        # Write all summary files in parallel
        summary_paths = await asyncio.gather(
            *(write_model_summary(result, temp_dir) for result in analysis_results)
        )

        # Create and write overview file
        overview_filename = f"a_{repo_info.name.split('/')[-1]}--overview"
        overview_path = temp_dir / f"{overview_filename}.md"

        today = datetime.now().strftime("%Y-%m-%d")
        github_repo_diff_link = (
            f"[{repo_info.name}]({repo_info.url}/compare/{first}...{last})"
        )
        overview_content = f"""*🔄 via [changes.py]({get_latest_github_commit_url(get_repo_info().name, "changes.py")}) - {today}*


_Returned after {completed_models}/{total_models} models finished (grace {grace_seconds}s)._

Changes to {github_repo_diff_link} From [{after}] To [{before}]

| Model | Analysis Duration (seconds) | Output Size (KB) |
|-------|---------------------------|-----------------|
"""

        for result in sorted(
            analysis_results,
            key=lambda x: x["analysis_duration"].total_seconds(),
            reverse=True,
        ):
            model_name = result["model_name"]
            safe_name = sanitize_model_name(model_name)
            duration = int(result["analysis_duration"].total_seconds())
            output_size = (
                len(result["diff_report"] + result["summary"]) / 1024
            )  # Convert to KB
            overview_content += f"| [{model_name}](#file-z_{safe_name}-md) | {duration} | {output_size:.1f} |\n"

        # Write overview file
        await asyncio.to_thread(overview_path.write_text, overview_content)

        files_to_gist = [overview_path] + list(summary_paths)

        if gist:
            # Create description using repo name and date range
            gist_description = f"changes - {repo_info.name} ({after} to {before})"
            # Clean up description by removing newlines and truncating if too long
            gist_description = gist_description.replace("\n", " ")[:100]
            await asyncio.to_thread(
                langchain_helper.to_gist_multiple,
                files_to_gist,
                description=gist_description,
            )
        else:
            print(overview_content)
            for result in analysis_results:
                model_name = result["model_name"]
                print(f"\n=== Analysis by {model_name} ===\n")
                print(result["diff_report"])


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
