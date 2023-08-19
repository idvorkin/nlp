#!python3

import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import Annotated, List

import openai
import pudb
import rich
import tiktoken
import typer
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import PythonREPL
from pydantic import BaseModel
from rich import print as rich_print
from rich.console import Console
from rich.text import Text
from typeguard import typechecked

from openai_wrapper import ask_gpt, choose_model, setup_gpt

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
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: Annotated[bool, typer.Option(prompt="use gpt4")] = False,
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


def openai_func(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


@app.command()
def journal_report(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
    u4: Annotated[bool, typer.Option("use gpt4")] = False,
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4, tokens)

    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))

    # Interesting we can specify in the prompt or in the "models" via text or type annotations
    class Person(BaseModel):
        Name: str
        Relationship: str
        Sentiment: str
        SummarizeInteraction: str

    class Category(str, Enum):
        Husband = ("husband",)
        Father = ("father",)
        Entertainer = ("entertainer",)
        PhysicalHealth = "physical_health"
        MentalHealth = "mental_health"
        Sleep = "sleep"
        Bicycle = "bicycle"
        Balloon = "balloon_artist"
        BeingAManager = "being_a_manager"
        BeingATechnologist = "being_a_technologist"

    class CategorySummary(BaseModel):
        TheCategory: Category
        Observations: str

    class Recommendation(BaseModel):
        ThingToDoDifferently: int
        ReframeToTellYourself: str
        PromptToUseDuringReflection: str
        ReasonIncluded: str

    class AssessmentWithReason(BaseModel):
        scale_1_to_10: int  # Todo see if can move scale to type annotation (condint
        reasoning_for_assessment: str

    class GetPychiatristReport(BaseModel):
        Date: datetime
        SummaryOfThePatientExperience: str
        Depression: AssessmentWithReason
        Anxiety: AssessmentWithReason
        Mania: AssessmentWithReason
        PromptsForCognativeReframes: List[str]
        PeopleInEntry: List[Person]
        Recommendations: List[Recommendation]
        CategorySummaries: List[CategorySummary]

    report = openai_func(GetPychiatristReport)

    system_prompt = f""" You are an expert psychologist who writes reports after reading patient's journal entries

You task it to write a report based on the journal entry that is going to be passed in

# Here are some facts to help you assess
{patient_facts()}

# Report

* Include 2-5 recommendations
* Don't include Category Summaries for Categories where you have no data
"""

    process_shared_app_options(ctx)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ]
    )
    model_name = "gpt-4" if u4 else "gpt-3.5-turbo"
    model = ChatOpenAI(
        model=model_name
    )  # Note, not yet using GPT-4 that could make the output much stronger ...
    ic(model_name)
    chain = (
        prompt
        | model.bind(function_call={"name": report["name"]}, functions=[report])
        | JsonOutputFunctionsParser()
    )
    # JsonKeyOutputFunctionsParser(key_name="jokes")
    response = chain.invoke({})
    print(json.dumps(response, indent=2))
    with open("out.json", "w") as f:
        json.dump(response, f)


if __name__ == "__main__":
    app()
