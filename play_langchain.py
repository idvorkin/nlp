#!python3


import os
import pickle
import sys
from typing import List
import subprocess

from langchain_core.messages.human import HumanMessage

import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from pathlib import Path
import json

console = Console()
app = typer.Typer()


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


def setup_secret():
    secret_file = Path.home() / "gits/igor2/secretBox.json"
    SECRETS = json.loads(secret_file.read_text())
    os.environ["OPENAI_API_KEY"] = SECRETS["openai"]


setup_secret()


chat = ChatOpenAI(temperature=0)
chat_model = chat


@logger.catch()
def app_wrap_loguru():
    app()


# Google search setup
# https://github.com/hwchase17/langchain/blob/d0c7f7c317ee595a421b19aa6d94672c96d7f42e/langchain/utilities/google_search.py#L9


@app.command()
def financial_agent(stock: str):
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools = load_tools(["bing-search"], llm=chat)
    # braveSearch = BraveSearch()
    # tools += [braveSearch]
    agent = initialize_agent(
        tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run(
        f"""
              What's the price outlook for : {stock}?, What are the top 3 reasons for the stock to go up?, What are the top 3 reasons for the stock to go down?

The output should be of the form:

Stock: XXX
Price: XXX
Price 1 year ago: XXX
Price 1 year ahead: XXX
Price goes up because:
- Point 1
- Point 2
- Point 3
Price goes down because:
- Point 1
- Point 2
- Point 3
"""
    )


def load_cached_prompt(prompt_name):
    from langchain import hub

    prompt_cache = os.path.expanduser("~/tmp/pickle/prompts")
    # if prompt_cache directory doesn't exist, create it
    if not os.path.exists(prompt_cache):
        os.makedirs(prompt_cache)
    prompt_maker_filename = f"{prompt_name.replace('/','_')}.pickle"
    prompt_maker_path = os.path.join(prompt_cache, prompt_maker_filename)

    if not os.path.exists(prompt_maker_path):
        prompt_maker_template = hub.pull(prompt_name)
        with open(prompt_maker_path, "wb") as f:
            pickle.dump(prompt_maker_template, f)
    else:
        with open(prompt_maker_path, "rb") as f:
            prompt_maker_template = pickle.load(f)

    return prompt_maker_template


@app.command()
def great_prompt(prompt):
    prompt_maker_template = load_cached_prompt("hardkothari/prompt-maker")
    model = ChatOpenAI(temperature=0.9)
    chain = prompt_maker_template | model
    result = chain.invoke({"lazy_prompt": prompt, "task": prompt})
    print(result.content)


@app.command()
def summarize():
    prompt_maker_template = load_cached_prompt("langchain-ai/chain-of-density:ba34ae10")
    user_text = "".join(sys.stdin.readlines())
    model = ChatOpenAI(temperature=0.9, model="gpt-4")
    chain = prompt_maker_template | model
    result = chain.invoke({"ARTICLE": user_text})
    print(result.content)


class GetHypotheticalQuestionsFromDoc(BaseModel):
    Questions: List[str]


@app.command()
def changes(revision_spec="HEAD@{7 days ago}"):
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

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )

    # Assuming the URL is in the form: https://github.com/idvorkin/bob or git@github.com:idvorkin/bob
    repo_url = result.stdout.strip()
    if repo_url.startswith("https"):
        base_path = repo_url.split("/")[-2] + "/" + repo_url.split("/")[-1]
    elif repo_url.startswith("git@"):
        base_path = repo_url.split(":")[1]
        base_path = base_path.replace(".git", "")

    print(f"# Changes in {base_path} from {revision_spec}")

    changed_files_command = ["git", "diff", "--name-only", revision_spec]
    ic(changed_files_command)
    result = subprocess.run(changed_files_command, capture_output=True, text=True)
    changed_files = result.stdout.split("\n")

    # Iterate through the list of changed files and generate summaries.
    file_diffs = []
    for file in changed_files:
        if file.strip() == "":
            continue
        file_path = Path(file)
        # Verify the file exists before proceeding.
        if not file_path.exists():
            ic(f"File {file} does not exist or has been deleted.")
            continue

        diff_command = ["git", "diff", revision_spec, "--", file]
        diff_result = subprocess.run(
            diff_command, capture_output=True, text=True, check=True
        )
        diff_content = diff_result.stdout
        file_diffs += [(file, diff_content)]

    # sort by length to do biggest changes first
    file_diffs.sort(key=lambda x: len(x[1]), reverse=True)
    for file, diff_content in file_diffs:
        ic(file)

        prompt1 = f"""Summarize the changes for {file}

## Instructions
    Have the first line be ### Filename on a single line
    Have second line be lines_added, lines_removed, lines change (but exclude changes in comments) on a single line

    Use a markdown list
    List the changes in order of impact, most impactful/major changes should go first.
    Exclude changes to imports
    Exclude changes to spelling, grammar or punctuation in the summary
    Exclude minor changes to wording, for example, exclude Changed "inprogress" to "in progress"
    When having larger changes include sub bullets.
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


@app.command()
def q_for_doc(questions: int = 10):
    get_questions = openai_wrapper.openai_func(GetHypotheticalQuestionsFromDoc)

    chain = (
        ChatPromptTemplate.from_template(
            "Generate a list of exactly {count} hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name).bind(
            functions=[get_questions], function_call={"name": get_questions["name"]}
        )
        | JsonKeyOutputFunctionsParser(key_name="Questions")
    )
    user_text = "".join(sys.stdin.readlines())
    r = chain.invoke({"doc": user_text, "count": questions})
    ic(r)


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
