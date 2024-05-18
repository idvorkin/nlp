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
    openai: bool = True,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
):
    llm = langchain_helper.get_model(
        openai=openai, google=google, claude=claude, llama=llama
    )
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


# Function to create the prompt


def prompt_report_from_diff(diff_summary, repo_url):
    instructions = f""" You are an expert programmer, who is charged with explaining code changes concisely. You always start with the why

You are given a diff output in a time range.

## Instructions

Begin by creating a TL;DR of the changes. With 1-2 lines per files. Order by the importance of the changes.

Then create a markdown table of contents for each file in the diff again ordered by importance, the table of contents will link to the details of the diff in the next section

<example>

E.g. for the file _d/foo.md, with 5 lines added, 3 lines removed, and 34 lines changed (excluding changes to comments)
E.g. for the file _d/bar.md, with 5 lines added, 3 lines removed, and 34 lines changed (excluding changes to comments)

Details:

* [_d/foo.md]({repo_url}/_d/foo.md): +5, -3, ~34
* [_d/bar.md]({repo_url}/_d/bar.md): +5, -3, ~34

</example>


After that go into more details on each diff.

<per_file_diff_instructions>

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

    <example>
    E.g. for the file _d/foo.md, with 5 lines added, 3 lines removed, and 34 lines changed (excluding changes to comments)

    #### _d/foo.md

    [_d/foo.md]({repo_url}/_d/foo.md): +5, -3, ~34

    TLDR: blah blah blah

    * Reason for change, chanage
        * Sub details of change
    </example>
</per_file_diff_instructions>





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

    diff_cmd = [
        "git",
        "diff",
        f"HEAD@{{{after}}}".replace(" ", "."),
        f"HEAD@{{{before}}}".replace(" ", "."),
    ]
    diff_output = subprocess.run(diff_cmd, capture_output=True, text=True).stdout
    report = (prompt_report_from_diff(diff_output, repo_url) | llm).invoke({}).content

    ic(report)
    print(report)
    ic(openai_wrapper.num_tokens_from_string(report))  # type:ignore
    ic((datetime.now() - start).total_seconds())
    output = f"""
{report}
"""

    output_file_path = Path(f"summary_{repo_name.split('/')[-1]}.md")
    with output_file_path.open("w", encoding="utf-8"):
        output_file_path.write_text(output)

    if gist:
        langchain_helper.to_gist(output_file_path)


#
#
#     file_diffs = await asyncio.gather(
#         *[
#             get_file_diff(file, first, last)
#             for file in changed_files
#             if not is_skip_file(file)
#         ]
#     )
#     # add some rate limiting
#     max_parallel = asyncio.Semaphore(100)
#
#     async def concurrent_llm_call(file, diff_content):
#         async with max_parallel:
#             ic(f"running on {file}")
#             return await (
#                 prompt_summarize_diff(
#                     file, diff_content, repo_path=repo_url, end_rev=last
#                 )
#                 | llm
#             ).ainvoke({})
#
#     ai_invoke_tasks = [
#         concurrent_llm_call(file, diff_content) for file, diff_content in file_diffs
#     ]
#
#     results = [result.content for result in await asyncio.gather(*ai_invoke_tasks)]
#
#     # I think this can be done by the reorder_diff_summary command
#     results.sort(key=lambda x: len(x), reverse=True)
#
#     unranked_diff_report = "\n\n___\n\n".join(results)
#
#     ic(unranked_diff_report)
#
#     # diff_report = (
#     #     (prompt_report_from_diff_summary(unranked_diff_report) | g_model)
#     #     .invoke({})
#     #     .content
#     # )
#     # print(diff_report)
#
#     ## Pre-ranked output
#     output = ""
#     github_repo_diff_link = f"[{repo_name}]({repo_url}/compare/{first}...{last})"
#     output = f"""
# ### Changes to {github_repo_diff_link} From [{after}] To [{before}]
# * Model: {langchain_helper.get_model_name(llm)}
# * Duration: {int((datetime.now() - start).total_seconds())} seconds
# * Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S") }
# ___
# {create_markdown_table_of_contents(changed_files)}
# ___
# {unranked_diff_report}
#     """
#     print(output)
#


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
