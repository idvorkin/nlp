#!python3


import os
import pickle
import sys
from typing import List


import openai_wrapper
import pudb
import typer
from icecream import ic
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console

console = Console()
app = typer.Typer()


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


openai_wrapper.setup_secret()


@logger.catch()
def app_wrap_loguru():
    app()


# Google search setup
# https://github.com/hwchase17/langchain/blob/d0c7f7c317ee595a421b19aa6d94672c96d7f42e/langchain/utilities/google_search.py#L9


@app.command()
def tell_me_a_joke(count=4):
    chat = ChatGoogleGenerativeAI(model="gemini-1.5.-pro-latest")
    template = ChatPromptTemplate.from_template(
        "Generate a list of exactly {count} joke about software engineers"
    )
    response = (template | chat).invoke({"count": count})
    ic(response)


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
