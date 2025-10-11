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
#     "langchain-xai",
#     "loguru",
#     "openai",
#     "pydantic",
#     "pudb",
#     "requests",
#     "rich",
#     "typer",
# ]
# ///

import os

# Suppress gRPC warnings from Google API - must be set before any gRPC imports
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

# Enable gRPC fork support to prevent deadlocks when using subprocess
# Must be called before any gRPC usage (langchain uses gRPC internally)
try:
    import grpc.experimental

    grpc.experimental.fork_support()
except (ImportError, AttributeError):
    # grpc module might not have fork_support in older versions
    pass

import asyncio
import subprocess
import tempfile
import uuid
import signal

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

# Semaphore limits for concurrent operations
MAX_CONCURRENT_FILE_DIFFS = 100  # Limit concurrent git diff operations
MAX_CONCURRENT_LLM_CALLS = 100  # Limit concurrent LLM API calls per model


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


async def run_subprocess_with_timeout(
    process, timeout: float, error_msg: str, verbose: bool = False
):
    """Run a subprocess with timeout and proper cleanup."""
    try:
        return await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        if verbose:
            print(f"Timeout: {error_msg}")

        # Try graceful termination first, then force kill
        try:
            process.terminate()  # Send SIGTERM
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Process didn't respond to SIGTERM, force kill with SIGKILL
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass  # Process is stuck, nothing more we can do
        except (ProcessLookupError, OSError):
            pass  # Process already dead

        raise RuntimeError(error_msg)


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
    try:
        stdout, _ = await run_subprocess_with_timeout(
            is_binary_cmd, 30.0, f"Binary check timed out for {file}", verbose
        )
    except RuntimeError:
        return file, f"Binary check timed out for {file}"
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
        stdout, _ = await run_subprocess_with_timeout(
            size_cmd, 10.0, f"Size check timed out for {file}", verbose
        )
        file_size_before = int(stdout.decode().strip()) if stdout else 0
    except Exception as e:
        if verbose:
            ic(f"Error getting previous size for {file}: {str(e)}")
        pass  # Continue even if we can't get file size

    # Use nbdiff for Jupyter notebooks to ignore outputs
    if verbose:
        ic(f"Starting diff process for {file}")

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
        try:
            stdout, _ = await run_subprocess_with_timeout(
                which_process, 5.0, f"which nbdiff timed out for {file}", verbose
            )
        except RuntimeError:
            stdout = None

        if stdout and stdout.decode().strip():
            if verbose:
                ic(f"Using nbdiff for {file}")

            nbdiff_process = await asyncio.create_subprocess_exec(
                "nbdiff",
                "--ignore-outputs",
                first_commit_hash,
                last_commit_hash,
                "--",
                file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout_diff, stderr = await run_subprocess_with_timeout(
                    nbdiff_process, 30.0, f"nbdiff timed out for {file}", verbose
                )
                use_git_diff = False
            except RuntimeError:
                # Timeout occurred, fall back to git diff
                if verbose:
                    ic(f"nbdiff timed out, falling back to git diff for {file}")
                use_git_diff = True
        else:
            if verbose:
                ic(f"nbdiff not found, falling back to git diff for {file}")
            use_git_diff = True
    else:
        use_git_diff = True

    if use_git_diff:
        if verbose:
            ic(f"Using git diff for {file}")

        git_diff_process = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            first_commit_hash,
            last_commit_hash,
            "--",
            file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout_diff, stderr = await run_subprocess_with_timeout(
                git_diff_process, 30.0, f"Git diff timed out for {file}", verbose
            )
        except RuntimeError:
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
    gist: bool = True,
    openai: bool = False,
    claude: bool = True,
    google: bool = False,  # Disabled - causes resource contention with other threads
    llama: bool = True,
    kimi: bool = True,
    grok4_fast: bool = True,
    fast: bool = typer.Option(
        False, help="Fast analysis using Llama and GPT-OSS models"
    ),
    only: str = None,
    md_only: bool = typer.Option(False, help="Only analyze markdown files (*.md)"),
    verbose: bool = False,
    gpt_oss: bool = False,
):
    # If fast is True, override other model selections to use llama and gpt_oss
    if fast:
        openai = False
        claude = False
        google = False
        llama = True
        gpt_oss = True
        grok4_fast = False
        if verbose:
            print("Fast mode: using Llama and GPT-OSS models for quick analysis")

    # If md_only is True, set the only pattern to *.md
    if md_only:
        only = "*.md"
        if verbose:
            print("Markdown-only mode: analyzing only *.md files")

    llms = langchain_helper.get_models(
        openai=openai,
        claude=claude,
        google=google,
        llama=llama,
        kimi=kimi,
        gpt_oss=gpt_oss,
        grok4_fast=grok4_fast,
    )

    # If no models are selected, provide a helpful error message
    if not llms:
        print("Error: No models selected. Please enable at least one model.")
        return

    achanges_params = llms, before, after, gist, only, verbose

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
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\n\nReceived interrupt signal. Cleaning up...")
            raise KeyboardInterrupt()

        # Store original handlers to restore them later
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

        try:
            asyncio.run(achanges(*achanges_params))
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            raise SystemExit(1)
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)


async def first_last_commit(before: str, after: str, verbose=False) -> Tuple[str, str]:
    if verbose:
        ic(f"git log --after={after} --before={before} --pretty=%H")

    # Execute the git log command with timeout (use exec not shell to prevent injection)
    process = await asyncio.create_subprocess_exec(
        "git",
        "log",
        f"--after={after}",
        f"--before={before}",
        "--pretty=%H",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await run_subprocess_with_timeout(
        process,
        30.0,
        f"Git log command timed out (after={after}, before={before})",
        verbose,
    )
    git_output = stdout.decode().strip().split("\n")

    if not git_output:
        print("No commits found for the specified date range.")
        return ("", "")

    # Extract the first and last commit hashes
    first_commit = git_output[-1]
    last_commit = git_output[0]

    # Get the diff before the last commit
    first_diff = await asyncio.create_subprocess_exec(
        "git",
        "log",
        f"{first_commit}^",
        "-1",
        "--pretty=%H",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await run_subprocess_with_timeout(
        first_diff,
        30.0,
        f"Git log command timed out for commit: {first_commit}^",
        verbose,
    )
    first_commit = stdout.decode().strip()

    if verbose:
        ic(first_commit, last_commit)
    return first_commit, last_commit


async def get_changed_files(first_commit, last_commit, verbose=False):
    # Execute the git diff command with timeout (use exec not shell to prevent injection)
    process = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "--name-only",
        first_commit,
        last_commit,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await run_subprocess_with_timeout(
        process,
        30.0,
        f"Git diff command timed out: {first_commit}...{last_commit}",
        verbose,
    )
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

When making file links, keep the _'s, here are valid examples:

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

IMPORTANT: Be concise. DO NOT output the entire report or individual file diffs. Only provide the summary and table of contents. Keep descriptions brief and to the point.

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

IMPORTANT: Be concise. DO NOT output the entire diff or code. Only summarize the changes. Keep descriptions brief and to the point.
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


def update_progress_table(
    model_names: List[str],
    model_states: dict,
    start_time: datetime,
    progress_log_path: str,
    detailed_logs: List[str] = None,
):
    """Update the progress table at the top of the log file"""
    try:
        # Count completed and total models
        completed = sum(
            1 for state in model_states.values() if state.get("status") == "✅ Done"
        )
        total = len(model_names)

        # Write new table
        with open(progress_log_path, "w") as f:
            f.write(
                f"# Changes Progress - Started at {start_time.strftime('%H:%M:%S')} - {completed}/{total} models complete\n\n"
            )
            f.write("| Model | Status | Files | Summary | Total Time |\n")
            f.write("|-------|--------|-------|---------|------------|\n")
            for model_name in model_names:
                state = model_states.get(model_name, {})
                status = state.get("status", "⏳ Waiting")
                files = state.get("files", "-")
                summary = state.get("summary", "-")
                total_time = state.get("total", "-")
                f.write(
                    f"| {model_name} | {status} | {files} | {summary} | {total_time} |\n"
                )

            # Write detailed logs in reverse order (newest first)
            if detailed_logs:
                f.write("\n---\n\n## Detailed Progress (newest first)\n\n")
                for log_entry in reversed(detailed_logs):
                    f.write(log_entry + "\n")

            f.flush()  # Ensure immediate write
    except Exception:
        pass  # Silently ignore errors


def init_progress_tracking(llms: List[BaseChatModel], total_start_time: datetime):
    """Initialize progress tracking for all models."""
    # Generate unique progress log file for this run (supports concurrent execution)
    progress_log_path = f"/tmp/changes_progress_{uuid.uuid4().hex[:8]}.log"

    # Create symlink to latest run
    latest_link = Path.home() / "tmp" / "changes.latest.txt"
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(progress_log_path)

    print(f"Progress log: {progress_log_path}")
    print(f"Latest link: {latest_link}")

    # Initialize model tracking and detailed logs
    model_names = [langchain_helper.get_model_name(llm) for llm in llms]
    model_states = {
        name: {"status": "⏳ Waiting", "files": "-", "summary": "-", "total": "-"}
        for name in model_names
    }
    detailed_logs = []  # In-memory buffer for detailed progress

    # Clear and initialize progress log with table
    update_progress_table(
        model_names, model_states, total_start_time, progress_log_path, detailed_logs
    )

    return progress_log_path, model_names, model_states, detailed_logs


async def fetch_git_changes(before: str, after: str, only: str, verbose: bool):
    """Fetch git commits and changed files."""
    # Get commit range
    first, last = await first_last_commit(before, after, verbose)

    # Get changed files
    changed_files = await get_changed_files(first, last, verbose)
    changed_files = [
        file for file in changed_files if not is_skip_file(file, only, verbose)
    ]

    return first, last, changed_files


async def process_file_diffs(
    changed_files: List[str], first: str, last: str, verbose: bool
):
    """Process all file diffs in parallel with semaphore limiting."""
    file_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILE_DIFFS)
    if verbose:
        ic(f"Created file semaphore with limit of {MAX_CONCURRENT_FILE_DIFFS}")

    async def get_file_diff_with_semaphore(file, first, last):
        diff_start = datetime.now()
        if verbose:
            ic(f"Waiting for file semaphore to process: {file}")
        async with file_semaphore:
            if verbose:
                ic(f"Acquired file semaphore for: {file}")
            result = await get_file_diff(file, first, last, verbose)
            if verbose:
                ic(f"Released file semaphore for: {file}")
        diff_end = datetime.now()
        diff_duration = (diff_end - diff_start).total_seconds()
        return result, diff_duration

    # Get all file diffs in parallel with semaphore
    git_diff_start = datetime.now()
    diff_results = await asyncio.gather(
        *[get_file_diff_with_semaphore(file, first, last) for file in changed_files]
    )
    git_diff_end = datetime.now()
    total_git_diff_duration = (git_diff_end - git_diff_start).total_seconds()

    # Extract file diffs and timings
    file_diffs = [(result[0], result[1]) for result, _ in diff_results]
    git_diff_timings = [
        {"file": result[0], "duration": duration}
        for (result, _), duration in zip(diff_results, [d for _, d in diff_results])
    ]

    return file_diffs, git_diff_timings, total_git_diff_duration


async def achanges(
    llms: List[BaseChatModel],
    before,
    after,
    gist,
    only: str = None,
    verbose: bool = False,
):
    """
    Analyze git changes using multiple LLM models in parallel.

    Main orchestration function that:
    1. Initializes progress tracking
    2. Fetches git changes
    3. Processes file diffs
    4. Analyzes with LLM models
    5. Generates and uploads reports
    """
    total_start_time = datetime.now()

    # Initialize progress tracking
    progress_log_path, model_names, model_states, detailed_logs = (
        init_progress_tracking(llms, total_start_time)
    )

    if verbose:
        ic("v 0.0.4")
    repo_info = get_repo_info(for_file_changes=True)
    repo_path = Path.cwd()

    # Fetch git changes
    first, last, changed_files = await fetch_git_changes(before, after, only, verbose)

    # Process file diffs
    file_diffs, git_diff_timings, total_git_diff_duration = await process_file_diffs(
        changed_files, first, last, verbose
    )

    # Process all models in parallel
    async def process_model(llm):
        model_start = datetime.now()
        model_name = langchain_helper.get_model_name(llm)

        # Update status to processing files
        model_states[model_name] = {
            "status": "📝 Files",
            "files": "0/0",
            "summary": "-",
            "total": "-",
        }
        update_progress_table(
            model_names,
            model_states,
            total_start_time,
            progress_log_path,
            detailed_logs,
        )

        max_parallel = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

        # Special handling for o4-mini model
        if isinstance(llm, ChatOpenAI) and llm.model_name == "o4-mini-2025-04-16":
            llm.model_kwargs = {}  # Clear any default parameters like temperature

        call_order = 0
        files_completed = 0
        call_order_lock = asyncio.Lock()
        model_process_start = datetime.now()
        total_files = len(file_diffs)

        async def concurrent_llm_call(file, diff_content):
            nonlocal call_order, files_completed
            model_name = langchain_helper.get_model_name(llm)
            queued_time = datetime.now()
            if verbose:
                ic(f"Waiting for max_parallel semaphore for {file} with {model_name}")
            async with max_parallel:
                # Record order when actually starting the call
                async with call_order_lock:
                    call_order += 1
                    my_order = call_order

                acquired_time = datetime.now()
                delay_to_start = (acquired_time - model_process_start).total_seconds()
                queue_wait_time = (acquired_time - queued_time).total_seconds()

                if verbose:
                    ic(f"Acquired max_parallel semaphore for {file} with {model_name}")
                    ic(f"++ LLM call start #{my_order}: {file} with {model_name}")

                # Log LLM call start
                call_start = datetime.now()
                start_elapsed = (call_start - total_start_time).total_seconds()
                content_size_k = len(diff_content) / 1000
                start_log = f"[{call_start.strftime('%H:%M:%S')}] [+{start_elapsed:.0f}s] {model_name}: START {file} ({content_size_k:.0f}K)"
                async with call_order_lock:
                    detailed_logs.append(start_log)
                    update_progress_table(
                        model_names,
                        model_states,
                        total_start_time,
                        progress_log_path,
                        detailed_logs,
                    )

                result = await (
                    prompt_summarize_diff(
                        file, diff_content, repo_path=repo_path, end_rev=last
                    )
                    | llm
                ).ainvoke({})
                call_end = datetime.now()
                call_duration = (call_end - call_start).total_seconds()

                # Update file completion counter
                async with call_order_lock:
                    files_completed += 1
                    elapsed = (datetime.now() - total_start_time).total_seconds()
                    model_states[model_name] = {
                        "status": "📝 Files",
                        "files": f"{files_completed}/{total_files}",
                        "summary": "-",
                        "total": f"{elapsed:.0f}s",
                    }

                    # Log individual file completion with timestamp and delta
                    remaining = total_files - files_completed
                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] [+{elapsed:.0f}s] {model_name}: DONE {file} ({call_duration:.1f}s) - {remaining} files remaining"
                    detailed_logs.append(log_msg)  # Add to in-memory buffer

                    update_progress_table(
                        model_names,
                        model_states,
                        total_start_time,
                        progress_log_path,
                        detailed_logs,
                    )

                if verbose:
                    ic(f"-- LLM call end: {file} with {model_name}")
                    ic(f"Released max_parallel semaphore for {file} with {model_name}")
                return (
                    result,
                    file,
                    call_duration,
                    my_order,
                    delay_to_start,
                    queue_wait_time,
                )

        # Run all file analyses for this model in parallel
        ai_invoke_tasks = [
            concurrent_llm_call(file, diff_content) for file, diff_content in file_diffs
        ]
        invoke_results = await asyncio.gather(*ai_invoke_tasks)

        # Extract timing data for each file
        file_timings = []
        results = []
        for (
            result,
            file,
            call_duration,
            order,
            delay_to_start,
            queue_wait_time,
        ) in invoke_results:
            results.append(result.content)
            file_timings.append(
                {
                    "file": file,
                    "duration": call_duration,
                    "order": order,
                    "delay_to_start": delay_to_start,
                    "queue_wait_time": queue_wait_time,
                }
            )

        results.sort(key=lambda x: len(x), reverse=True)
        code_based_diff_report = "\n\n___\n\n".join(results)

        model_name = langchain_helper.get_model_name(llm)
        # Get summary for this model
        if verbose:
            ic(f"++ LLM summary call start with {model_name}")

        # Update status to summary
        elapsed = (datetime.now() - total_start_time).total_seconds()
        model_states[model_name] = {
            "status": "📊 Summary",
            "files": f"{total_files}/{total_files}",
            "summary": "⏳",
            "total": f"{elapsed:.0f}s",
        }

        # Log summary start with timestamp and delta
        summary_content_size_k = len(code_based_diff_report) / 1000
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] [+{elapsed:.0f}s] {model_name}: Starting summary ({summary_content_size_k:.0f}K)..."
        detailed_logs.append(log_msg)  # Add to in-memory buffer

        update_progress_table(
            model_names,
            model_states,
            total_start_time,
            progress_log_path,
            detailed_logs,
        )

        summary_start = datetime.now()
        summary_all_diffs = await (
            prompt_summarize_diff_summaries(code_based_diff_report) | llm
        ).ainvoke({})
        summary_end = datetime.now()
        summary_duration = (summary_end - summary_start).total_seconds()

        # Update status to complete
        elapsed = (datetime.now() - total_start_time).total_seconds()
        model_states[model_name] = {
            "status": "✅ Done",
            "files": f"{total_files}/{total_files}",
            "summary": f"{summary_duration:.1f}s",
            "total": f"{elapsed:.0f}s",
        }

        # Log summary completion with timestamp and delta
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] [+{elapsed:.0f}s] {model_name}: Summary complete ({summary_duration:.1f}s)"
        detailed_logs.append(log_msg)  # Add to in-memory buffer

        update_progress_table(
            model_names,
            model_states,
            total_start_time,
            progress_log_path,
            detailed_logs,
        )

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
            "file_timings": file_timings,
            "summary_duration": summary_duration,
        }

    # Run all models in parallel
    models_start_time = datetime.now()
    analysis_results = await asyncio.gather(*(process_model(llm) for llm in llms))
    models_end_time = datetime.now()
    total_models_duration = (models_end_time - models_start_time).total_seconds()

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
        file_write_start = datetime.now()
        summary_paths = await asyncio.gather(
            *(write_model_summary(result, temp_dir) for result in analysis_results)
        )
        file_write_end = datetime.now()
        file_write_duration = (file_write_end - file_write_start).total_seconds()

        # Create and write overview file
        overview_filename = f"a_{repo_info.name.split('/')[-1]}--overview"
        overview_path = temp_dir / f"{overview_filename}.md"

        today = datetime.now().strftime("%Y-%m-%d")
        github_repo_diff_link = (
            f"[{repo_info.name}]({repo_info.url}/compare/{first}...{last})"
        )
        overview_content = f"""*🔄 via [changes.py]({get_latest_github_commit_url(get_repo_info().name, "changes.py")}) - {today}*

Changes to {github_repo_diff_link} From [{after}] To [{before}]

[Detailed Timing Debug Info](#file-zzz_timing_debug-md)

| Model | Analysis Duration (seconds) | Output Size (KB) |
|-------|---------------------------|-----------------|
"""

        # Create detailed timing debug file
        total_current_duration = (datetime.now() - total_start_time).total_seconds()
        gap_time = (
            total_current_duration
            - total_git_diff_duration
            - total_models_duration
            - file_write_duration
        )

        debug_content = f"""# Detailed LLM Call Timing Debug

Changes to {github_repo_diff_link} From [{after}] To [{before}]

## Overall Timing Summary

| Phase | Duration (s) | % of Total |
|-------|--------------|------------|
| Git Diff Operations | {total_git_diff_duration:.2f} | {(total_git_diff_duration / total_current_duration * 100):.1f}% |
| All Models Processing | {total_models_duration:.2f} | {(total_models_duration / total_current_duration * 100):.1f}% |
| File Writing | {file_write_duration:.2f} | {(file_write_duration / total_current_duration * 100):.1f}% |
| Other/Overhead | {gap_time:.2f} | {(gap_time / total_current_duration * 100):.1f}% |
| **Total (so far)** | **{total_current_duration:.2f}** | **100%** |

**Note:** Gist upload timing will be added after completion.

---

## Git Diff Timing

**Total Git Diff Duration:** {total_git_diff_duration:.2f} seconds

### Per-File Git Diff Timings (sorted by duration, longest first)

| File | Duration (s) |
|------|--------------|
"""

        # Add git diff timings
        sorted_git_timings = sorted(
            git_diff_timings, key=lambda x: x["duration"], reverse=True
        )
        for timing in sorted_git_timings:
            debug_content += f"| {timing['file']} | {timing['duration']:.2f} |\n"

        debug_content += "\n---\n\n## Table of Contents\n\n"

        # Build ToC
        for result in sorted(
            analysis_results,
            key=lambda x: x["analysis_duration"].total_seconds(),
            reverse=True,
        ):
            model_name = result["model_name"]
            safe_name = sanitize_model_name(model_name)
            debug_content += f"- [{model_name}](#timing-{safe_name})\n"

        debug_content += "\n---\n"

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

            # Add to debug content
            debug_content += f'\n## <a id="timing-{safe_name}"></a>{model_name}\n\n'
            debug_content += f"**Total Duration:** {duration} seconds\n"
            debug_content += f"**Summary Call Duration:** {result['summary_duration']:.2f} seconds\n\n"
            debug_content += (
                "### Per-File Timings (sorted by duration, longest first)\n\n"
            )
            debug_content += "| Order | File | Duration (s) | Delay to Start (s) | Queue Wait (s) |\n"
            debug_content += (
                "|-------|------|--------------|-------------------|----------------|\n"
            )

            # Sort file timings by duration, longest first
            sorted_by_duration = sorted(
                result["file_timings"], key=lambda x: x["duration"], reverse=True
            )

            for timing in sorted_by_duration:
                debug_content += f"| {timing['order']} | {timing['file']} | {timing['duration']:.2f} | {timing['delay_to_start']:.2f} | {timing['queue_wait_time']:.2f} |\n"

        # Write overview file
        await asyncio.to_thread(overview_path.write_text, overview_content)

        # Write debug timing file
        debug_path = temp_dir / "zzz_timing_debug.md"
        await asyncio.to_thread(debug_path.write_text, debug_content)

        files_to_gist = [overview_path] + list(summary_paths) + [debug_path]

        if gist:
            # Create description using repo name and date range
            gist_description = f"changes - {repo_info.name} ({after} to {before})"
            # Clean up description by removing newlines and truncating if too long
            gist_description = gist_description.replace("\n", " ")[:100]
            gist_upload_start = datetime.now()
            await asyncio.to_thread(
                langchain_helper.to_gist_multiple,
                files_to_gist,
                description=gist_description,
            )
            gist_upload_end = datetime.now()
            gist_upload_duration = (gist_upload_end - gist_upload_start).total_seconds()
        else:
            gist_upload_duration = 0
            print(overview_content)
            for result in analysis_results:
                model_name = result["model_name"]
                print(f"\n=== Analysis by {model_name} ===\n")
                print(result["diff_report"])

    total_end_time = datetime.now()
    total_duration = (total_end_time - total_start_time).total_seconds()

    if verbose:
        ic(f"Total end-to-end duration: {total_duration:.2f}s")
        ic(f"Git diff duration: {total_git_diff_duration:.2f}s")
        ic(f"Models processing duration: {total_models_duration:.2f}s")
        ic(f"File writing duration: {file_write_duration:.2f}s")
        ic(f"Gist upload duration: {gist_upload_duration:.2f}s")


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
