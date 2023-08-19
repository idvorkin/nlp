#!python3

import os
from pydantic import BaseModel
from datetime import datetime
import openai
import json
from icecream import ic
import typer
import sys
from rich import print as rich_print
from rich.console import Console
from rich.text import Text
import rich
import re
from typeguard import typechecked
import tiktoken
import time
from typing import List
from openai_wrapper import choose_model, setup_gpt, ask_gpt
import pudb
from typing_extensions import Annotated

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import PythonREPL
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


import signal

console = Console()

# By default, when you hit C-C in a pipe, the pipe is stopped
# with this, pipe continues
def keep_pipe_alive_on_control_c(sig, frame):
    sys.stdout.write(
        "\nInterrupted with Control+C, but I'm still writing to stdout...\n"
    )
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, keep_pipe_alive_on_control_c)

original_print = print
is_from_console = False


gpt_model = setup_gpt()
app = typer.Typer()

# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    attach: bool = Annotated[bool, typer.Option(prompt="Attach to existing process")],
):
    ctx.obj = SimpleNamespace(attach=attach)


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


# GPT performs poorly with trailing spaces (wow this function was writting by gpt)
def remove_trailing_spaces(str):
    return re.sub(r"\s+$", "", str)


@app.command()
def group(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    prompt_to_gpt = f"""Group the following:
-----
{user_text}

"""
    # base_query_from_dict(locals())


def patient_facts():
    return """
* Kiro is a co-worker
* Zach, born in 2010 is son
* Amelia, born in 2014 is daughter
* Tori is wife
* Physical Habits is the same as physical health and exercisies
* Bubbles are a joy activity
* Turkish Getups (TGU) is about physical habits
* Swings refers to Kettle Bell Swings
* Treadmills are about physical health
* 750words is journalling
* I work as an engineering manager (EM) in a tech company
* A refresher is a synonym for going to the gym
"""


@app.command()
def life_group(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    prompt_to_gpt = f"""

You will be grouping elements. I'll provide group categories, facts, and then group the items.


* Try not to put items into multiple categories;
* Use markdown for category titles.
* Use facts to help you understand the grouping, not to be included into the output

# Group categories

<!-- prettier-ignore-start -->
<!-- vim-markdown-toc GFM -->

- [Work]
    - [Individual Contributor/Tech]
    - [Manager]
- [Friends]
- [Family]
    - [Tori](#tori)
    - [Zach](#zach)
    - [Amelia](#amelia)
- [Magic]
    - [Performing](#performing)
    - [Practice](#practice)
    - [General Magic](#general-magic)
- [Tech Guru]
    - [Blogging](#blogging)
    - [Programming](#programming)
- [Identity Health]
    - [Biking](#biking)
    - [Ballooning](#ballooning)
    - [Joy Activities](#joy-activities)
- [Motivation]
- [Emotional Health]
    - [Meditation](#meditation)
    - [750 words](#750-words)
    - [Avoid Procrastination](#avoid-procrastination)
- [Physical Health]
    - [Exercise](#exercise)
    - [Diet](#diet)
    - [Sleep](#sleep)
- [Inner Peace]
    - [General](#general)
    - [Work](#work)
    - [Family](#family)
- [Things not mentioned above]

<!-- vim-markdown-toc GFM -->
<!-- prettier-ignore-end -->

# Facts

{patient_facts()}

# Items

{user_text}
"""

    #  base_query_from_dict(locals())


@app.command()
def mood_old(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4, tokens)

    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = f""" You are an expert psychologist who writes reports
    after reading patient's journal entries

You task it to write a report based on the passed in journal entry.

The reports include the entry date in the summary,
and a 1-10 scale rating of anxiety, depression, and mania.

Summary:
- Includes entry date
- Provides rating of depression anxiety and mania

Depression Rating:
- Uses a 1-10 scale
- 0 represents mild depression
- 10 represents hypomania
- Provides justification for rating

Anxiety Rating:
- Uses a 1-10 scale
- 10 signifies high anxiety
- Provides justification for rating

Mania Rating:
- Uses a 1-10 scale
- 10 signifies mania
- 5 signifies hypomania
- Provides justification for rating


# Here are some facts to help you assess
{patient_facts()}

# Report
<!-- prettier-ignore-start -->
<!-- vim-markdown-toc GFM -->

- [Summary, assessing mania, depression, and anxiety]
- [Summary of patient experience]
- [Patient Recommendations]
- [Journal prompts to support cognitive reframes, with concrete examples]
- [People and relationships, using names when possible]

<!-- prettier-ignore-end -->
<!-- vim-markdown-toc GFM -->
"""

    prompt_to_gpt = user_text
    # base_query_from_dict(locals())


def openai_func(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


@app.command()
def mood(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4, tokens)

    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = f""" You are an expert psychologist who writes reports after reading patient's journal entries

You task it to write a report based on the passed in journal entry.

# Here are some facts to help you assess
{patient_facts()}

# Report

* Include 2-5 recommendations
"""

    class Person(BaseModel):
        name: str
        relationship: str
        interaction: str
        sentiment: str
        reason_for_inclusion: str

    class Recommendation(BaseModel):
        Recommendation: int  # Todo see if can move scale to type annotation (condint
        ReframeToUse: str
        PromptToUse: str
        reason_for_inclusion: str

    class AssessmentWithReason(BaseModel):
        scale_1_to_10: int  # Todo see if can move scale to type annotation (condint
        reasoning_for_assessment: str

    class GetPychiatristReprort(BaseModel):
        Date: datetime
        SummaryOfThePatientExperience: str
        Depression: AssessmentWithReason
        Anxiety: AssessmentWithReason
        Mania: AssessmentWithReason
        PromptsForCognativeReframes: List[str]
        PeopleInEntry: List[Person]
        Recommendations: List[Recommendation]

    report = openai_func(GetPychiatristReprort)

    process_shared_app_options(ctx)
    model = ChatOpenAI()
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ]
    )
    chain = prompt | model.bind(functions=[report]) | JsonOutputFunctionsParser()
    # JsonKeyOutputFunctionsParser(key_name="jokes")
    response = chain.invoke({"topic": "Joke", "count": 2})
    #  rich_print(response)
    print(response)


if __name__ == "__main__":
    app()
