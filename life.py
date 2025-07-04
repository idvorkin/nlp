#!python3

import asyncio
import glob
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Annotated, Dict, List

from pydantic import BaseModel, field_validator, ValidationError

# from pydantic import BaseModel, field_validator
import typer
from icecream import ic
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import track
from langchain_community.vectorstores import FAISS
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


import igor_journal
from openai_wrapper import num_tokens_from_string, setup_gpt, setup_secret
import langchain_helper
from pathlib import Path

setup_secret()

console = Console()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# By default, when you hit C-C in a pipe, the pipe is stopped
# with this, pipe continues
def keep_pipe_alive_on_control_c(signum, frame):
    del signum, frame  # unused variables
    sys.stdout.write(
        "\nInterrupted with Control+C, but I'm still writing to stdout...\n"
    )
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, keep_pipe_alive_on_control_c)

original_print = print
is_from_console = False


gpt_model = setup_gpt()
app = typer.Typer(no_args_is_help=True)


# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    u4: Annotated[bool, typer.Option] = typer.Option(False),
):
    ctx.obj = SimpleNamespace(u4=u4)


def process_shared_app_options(ctx: typer.Context):
    return ctx


# GPT performs poorly with trailing spaces (wow this function was writting by gpt)
def remove_trailing_spaces(str):
    return re.sub(r"\s+$", "", str)


@app.command()
def group2(
    ctx: typer.Context,
    markdown: Annotated[bool, typer.Option()] = False,
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))

    valence = "positive"
    system_prompt = f"""

You are given a csv of waht makes a person {valence} every day. Please give a monthly summary of what makes the person {valence}, with relative weigths of what makes them {valence}.

* Output in markdown, for each thing, include 3-6 sub bullets
* Provide data for each month
* Do not stop summarizing until you've summarized every month in the input document

E.g.

### January 2019

* Physical Health (40%) Upset because injured
    * Skipped going to gym
    * Hurt back doing deadlifts


     """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )
    model = langchain_helper.get_model(openai=True)
    ic(langchain_helper.get_model_name(model))
    ic(num_tokens_from_string(user_text))

    start = time.time()
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({})
    if markdown:
        console = Console()
        md = Markdown(response)
        console.print(md)
    else:
        print(response)
    total = time.time() - start
    ic(f"Total time: {total} seconds")


@app.command()
def summary_days(
    ctx: typer.Context,
    markdown: Annotated[bool, typer.Option()] = True,
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))

    system_prompt = f"""You read journal entries and help summarize the entries into point form based on categories.  Output the category headers as markdown, and list the line items as list eelemnts below. Eg.

# Grouping A
* line 1
* line 2

If there are changes you recommend, include them: E.g.

* STOP:  Doing handstands they hurt your shoulders
* START: Do card tricks

IF possible, categories should match the following

- [Dealer of smiles and wonder](#dealer-of-smiles-and-wonder)
- [Mostly car free spirit](#mostly-car-free-spirit)
- [Disciple of the 7 habits of highly effective people](#disciple-of-the-7-habits-of-highly-effective-people)
- [Fit fellow](#fit-fellow)
- [Emotionally healthy human](#emotionally-healthy-human)
- [Husband to Tori - his life long partner](#husband-to-tori---his-life-long-partner)
- [Technologist](#technologist)
- [Professional](#professional)
- [Family man](#family-man)
- [Father to Amelia - an incredible girl](#father-to-amelia---an-incredible-girl)
- [Father to Zach - a wonderful boy](#father-to-zach---a-wonderful-boy)

<patient_facts>
{patient_facts()}
</patient_facts>

     """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )
    model = langchain_helper.get_model(openai=True)
    ic(langchain_helper.get_model_name(model))
    ic(num_tokens_from_string(user_text))

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({})
    if markdown:
        console = Console()
        md = Markdown(response)
        console.print(md)
    else:
        print(response)


@app.command()
def group(
    ctx: typer.Context,
    markdown: Annotated[bool, typer.Option()] = True,
    claude: Annotated[
        bool, typer.Option()
    ] = False,  # use OpenAI sas it has a much larget context window
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))

    system_prompt = """You help group similar items into categories.  Exclude any linnes that are markdown headers. Output the category headers as markdown, and list the line items as list eelemnts below. Eg.

# Grouping A
* line 1
* line 2

IF possible, categories should match the following

- [Dealer of smiles and wonder](#dealer-of-smiles-and-wonder)
- [Mostly car free spirit](#mostly-car-free-spirit)
- [Disciple of the 7 habits of highly effective people](#disciple-of-the-7-habits-of-highly-effective-people)
- [Fit fellow](#fit-fellow)
- [Emotionally healthy human](#emotionally-healthy-human)
- [Husband to Tori - his life long partner](#husband-to-tori---his-life-long-partner)
- [Technologist](#technologist)
- [Professional](#professional)
- [Family man](#family-man)
- [Father to Amelia - an incredible girl](#father-to-amelia---an-incredible-girl)
- [Father to Zach - a wonderful boy](#father-to-zach---a-wonderful-boy)

If there are multiple instances of an item start it wtih with the number of times e.g.

3 x Eating Chips

     """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )

    model = (
        langchain_helper.get_model(claude=True)
        if claude
        else langchain_helper.get_model(openai=True)
    )
    ic(langchain_helper.get_model_name(model))
    ic(num_tokens_from_string(user_text))

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({})
    if markdown:
        console = Console()
        md = Markdown(response)
        console.print(md)
    else:
        print(response)


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
* PSC =>  Performance Summary Cycle (Writing performance reviews)
"""


# Interesting we can specify in the prompt or in the "models" via text or type annotations
class Person(BaseModel):
    Name: str
    Relationship: str
    Sentiment: str
    SummarizeInteraction: str


class Category(str, Enum):
    Husband = "husband"
    Father = "father"
    Entertainer = "entertainer"
    PhysicalHealth = "physical_health"
    MentalHealth = "mental_health"
    Sleep = "sleep"
    Bicycle = "bicycle"
    Balloon = "balloon_artist"
    BeingAManager = "being_a_manager"
    BeingATechnologist = "being_a_technologist"
    Unknown = "unknown"


class CategorySummary(BaseModel):
    TheCategory: Category
    Observations: List[str]

    @field_validator("TheCategory", mode="before")
    @classmethod
    def parse_category(cls, value):
        # Check if value is one of the enum's values
        for member in Category:
            if member.value == value:
                return member
        # If not found, use ic to debug and return Unknown
        ic(value)
        return Category.Unknown


class Recommendation(BaseModel):
    ReasonIncluded: str
    ThingToDoDifferently: str
    ReframeToTellYourself: str
    PromptToUseDuringReflection: str


class AssessmentWithReason(BaseModel):
    reasoning_for_assessment: str
    scale_1_to_10: int  # Todo see if can move scale to type annotation (condint


class Causes(BaseModel):
    reason: str
    emotion: str
    scale_1_to_10: int  # Todo see if can move scale to type annotation (condint


class GetPychiatristReport(BaseModel):
    Date: datetime
    DoctorName: str
    PointFormSummaryOfEntry: List[str]
    MentionedDays: List[str]  # New field to track all days mentioned in the journal entry
    Depression: AssessmentWithReason
    Anxiety: AssessmentWithReason
    Mania: AssessmentWithReason
    Happiness: AssessmentWithReason
    PostiveEmotionCause: List[Causes]
    NegativeEmotionCause: List[Causes]
    Satisfication: AssessmentWithReason
    CategorySummaries: List[CategorySummary]
    PromptsForCognativeReframes: List[str]
    PeopleInEntry: List[Person]
    Recommendations: List[Recommendation]

    @field_validator("Date", mode="before")
    @classmethod
    def parse_date(cls, value):
        date_formats = [
            "%m-%d-%Y",
            "%Y/%m/%d",
            "%d %b, %Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        for date_format in date_formats:
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                continue
        raise ValueError(f"Date {value} is not a valid date format")


def openai_func(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


@app.command()
def journal_report(
    journal_for: str = typer.Argument(
        datetime.now().date(), help="Pass a date or int for days ago"
    ),
    launch_fx: Annotated[bool, typer.Option()] = True,
    days: int = 1,
    file: Annotated[
        Path, typer.Option(help="Path to journal file to use instead of ij command")
    ] = None,
):
    asyncio.run(async_journal_report(journal_for, launch_fx, days, file))


def spark_df(df):
    from rich import print
    from rich.table import Table
    from sparklines import sparklines

    rich_table = Table()
    rich_table.add_column("Category")
    for col in df.columns:
        clean = df[col]
        spark = sparklines(clean, minimum=0, maximum=10)
        spark_str = "".join(spark)
        # reverse the string
        col = col.ljust(max([len(c) for c in df.columns]) + 1)
        print(f"{col}[blue]{spark_str}[/blue]")


@app.command()
def stats(
    days: int = 7,
    journal_for: str = typer.Argument(
        datetime.now().date(), help="Pass a date or int for days ago"
    ),
):
    cEntries = 0
    for i in range(days):
        day = date.fromisoformat(journal_for) - timedelta(days=i)
        try:
            entry = igor_journal.JournalEntry(day)
            if not entry.is_valid():
                continue
            cEntries += 1
        except FileNotFoundError:
            continue
    ic(cEntries)


@app.command()
def journal_for_year():
    asyncio.run(async_journal_for_year())


@app.command()
def insights():
    get_reports()


tmp = Path("~/tmp")
faiss_path_igor_journal = Path("~/tmp/igor_journal_faiss")


def journal_entry_to_document(entry: igor_journal.JournalEntry):
    metadata = {"date": str(entry.date)}
    return Document(page_content="\n".join(entry.body()), metadata=metadata)


@app.command()
def build_index_for_journal():
    valid_dates = igor_journal.all_entries()
    journal_entries = [igor_journal.JournalEntry(date) for date in valid_dates]
    documents = [journal_entry_to_document(j) for j in journal_entries]

    search_index = FAISS.from_documents(documents, embeddings)
    search_index.save_local(str(faiss_path_igor_journal))


@app.command()
def closest_journal_entries_stdin(
    count: int = 15,
):
    index = FAISS.load_local(
        str(faiss_path_igor_journal), embeddings, allow_dangerous_deserialization=True
    )

    str_stdin = "".join(sys.stdin.readlines())
    nearest_documents = index.similarity_search_with_score(str_stdin, k=count)
    for f, score in nearest_documents:
        ic(f.metadata["date"], score)


@app.command()
def closest_journal_entries(
    journal_for: Annotated[
        str, typer.Argument(help="Pass a date or int for days ago")
    ] = str(datetime.now().date()),
    close: Annotated[
        bool,
        typer.Option(
            help="Keep going back in days till you find the closest valid one"
        ),
    ] = True,
    count: int = 15,
):
    date_journal_for = igor_journal.cli_date_to_entry_date(journal_for, close)
    ic(date_journal_for)
    entry = igor_journal.JournalEntry(date_journal_for)

    if not entry.is_valid():
        raise FileNotFoundError(f"No Entry for {date_journal_for} ")

    index = FAISS.load_local(
        str(faiss_path_igor_journal), embeddings, allow_dangerous_deserialization=True
    )

    nearest_documents = index.similarity_search_with_score(
        "\n".join(entry.body()), k=count
    )
    for f, score in nearest_documents:
        ic(f.metadata["date"], score)


def get_reports():
    path_reports = glob.glob(
        os.path.expanduser("~/tmp/journal_report/*4-*-preview.json")
    )
    reports = []
    validation_errors = {}
    for path_report in path_reports:
        text_report = open(path_report, "r").read()
        try:
            # Odd, often I don't have recommendations, lets manually add them to
            # avoid a validation error
            json_report = json.loads(text_report)
            if "CategorySummaries" not in json_report:
                json_report["CategorySummaries"] = []
            # report = GetPychiatristReport.model_validate(json_report)
            report = GetPychiatristReport.model_validate(json_report)
            reports += [report]
        except ValidationError as ve:
            ic(f"Validation Error {path_report}")
            for e in ve.errors():
                error = e["type"], e["loc"]
                validation_errors[error] = validation_errors.get(error, 0) + 1
                # ic(error)
        except Exception as e:
            ic("Exception", path_report, e)

    ic(validation_errors)
    ic(len(path_reports), len(reports))
    return reports


def get_reports_cached():
    # load from pickle file
    return pickle.load(open(f"{tmp}/reports.pkl", "rb"))


# pickle.dump(get_reports(), open(f"{tmp}/reports.pkl", "wb"))
# reports =  load_all_reports()


def journal_report_path(date: str, model: str):
    return os.path.expanduser(
        f"~/tmp/journal_report/{date}_{model}.json".replace(" ", "_").lower()
    )


async def async_journal_for_year():
    for entry_date in igor_journal.all_entries():
        ic(entry_date)
        model = langchain_helper.get_model(openai=True)

        journal_path = journal_report_path(
            date=entry_date, model=langchain_helper.get_model_name(model)
        )
        if os.path.exists(journal_path):
            ic("Exists", journal_path)
            continue
        try:
            await async_journal_report(journal_for=entry_date, launch_fx=False, days=1)
        except Exception as e:
            # swallow exeception and keep going
            ic(entry_date, e)


async def async_journal_report(journal_for, launch_fx, days, file=None):
    # Get my closest journal for the day:

    if file and file.exists():
        # Read from the provided file
        user_text = file.read_text()
    else:
        # ij is the name for Igor Journal
        completed_process = subprocess.run(
            f"ij body {journal_for} --close --days={days}",
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )
        user_text = completed_process.stdout

    # Escape any curly braces in the user text to prevent f-string parsing errors
    user_text = user_text.replace("{", "{{").replace("}", "}}")

    # remove_trailing_spaces("".join(sys.stdin.readlines()))

    system_prompt = f""" You are an expert psychologist named Dr {{model}} who writes reports after reading patient's journal entries

You task it to write a report based on the journal entry that is going to be passed in

# Here are some facts to help you assess
{patient_facts()}

# Report

* Include 2-5 recommendations
* Don't include Category Summaries for Categories where you have no data
* Identify and list all dates mentioned in the journal entry (in YYYY-MM-DD format when possible)
"""

    start = time.time()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_text),
        ],
    )

    model = langchain_helper.get_model(google=True)
    model_name = langchain_helper.get_model_name(model)
    ic(model_name)
    # HACK need to pass function calling, tillt his is fixed.
    # https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D0.3.0
    chain = prompt | model.with_structured_output(
        GetPychiatristReport
    )

    corourtine = chain.ainvoke({"model": model_name})
    do_invoke = asyncio.create_task(corourtine)

    if launch_fx:
        for _ in track(range(30), description="30 seconds"):
            if do_invoke.done():
                break
            await asyncio.sleep(1)  # Simulate work being done

    # should now be done!
    pych_report: GetPychiatristReport = await do_invoke  # noqa

    # Ensure tmp/journal_report directory exists before writing files
    latest_path = Path.home() / "tmp/journal_report/latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(pych_report.model_dump_json(indent=2))

    report_date = pych_report.Date.strftime("%Y-%m-%d")

    perma_path = Path(journal_report_path(report_date, model=model_name))
    perma_path.parent.mkdir(parents=True, exist_ok=True)
    perma_path.write_text(pych_report.model_dump_json(indent=2))
    print(pych_report.model_dump_json())
    print(perma_path)

    total = time.time() - start
    print(f"Total time: {total} seconds")
    if launch_fx:
        subprocess.run(f"fx {perma_path}", shell=True)


def serialize_model():
    ic("Hello")
    print("hello")


def to_people_sentiment_dict(r: GetPychiatristReport):
    row: Dict = {"date": r.Date}
    for p in r.PeopleInEntry:
        sentiment = p.Sentiment.lower()
        if sentiment in ["not mentioned", "unmentioned"]:
            continue
        if sentiment == "concerned":
            sentiment = "concern"
        row[p.Name.lower()] = sentiment

    return row


if __name__ == "__main__":
    app()
