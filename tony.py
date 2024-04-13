#!python3


import asyncio

from langchain_core import messages
from langchain_core.pydantic_v1 import BaseModel

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
import httpx
from icecream import ic
from datetime import datetime

console = Console()
app = typer.Typer()


@logger.catch()
def app_wrap_loguru():
    app()


def prompt_transcribe_call(transcript):
    instructions = """
You are an AI assistant transcriber. Below is a  conversation with the assisatant (Tony) and a user (Igor)  that may capture todos and reminders
Parse them with a function call

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=transcript),
        ]
    )


class CallSummary(BaseModel):
    Notes: str
    Reminders: list[str]
    JournalEntry: list[str]
    CompletedHabits: list[str]
    CallSummary: list[str]


class Call(BaseModel):
    Caller: str
    Transcript: str
    Start: datetime


def parse_call(call) -> Call:
    customer = ""
    if "customer" in call:
        customer = call["customer"]["number"]

    return Call(
        Caller=customer,
        Transcript=call.get("transcript", ""),
        Start=datetime.strptime(call["createdAt"], "%Y-%m-%dT%H:%M:%S.%fZ"),
    )


def vapi_calls() -> list[Call]:
    # list all calls from VAPI
    # help:  https://api.vapi.ai/api#/Calls/CallController_findAll
    import os

    headers = {"authorization": f"{os.environ['VAPI_API_KEY']}"}
    # future add createdAtGe
    return [
        parse_call(c)
        for c in httpx.get("https://api.vapi.ai/call", headers=headers).json()
    ]


@app.command()
def calls():
    calls = vapi_calls()
    for call in calls:
        ic(call.Caller, call.Start.strftime("%Y-%m-%d %H:%M"), len(call.Transcript))
    ic(len(calls))


@app.command()
def parse_calls(
    trace: bool = False,
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_parse_calls())
    )


async def a_parse_calls():
    llms = [
        langchain_helper.get_model(openai=True),
        langchain_helper.get_model(claude=True),
    ]
    # google = langchain_helper.get_model(google=True)

    async def transcribe_call(user_text, llm):
        description = (
            await (prompt_transcribe_call(user_text) | llm)
            .with_structured_output(CallSummary)
            .ainvoke({})
        )
        return description.content, llm

    calls = vapi_calls()
    # first call of length > 1000
    call = next((c for c in calls if len(c.Transcript) > 1000))

    describe_diff_tasks = [transcribe_call(call.Transcript, llm) for llm in llms]
    describe_diffs = [result for result in await asyncio.gather(*describe_diff_tasks)]

    for description, llm in describe_diffs:
        print(f"# -- model: {langchain_helper.get_model_name(llm)} --")
        print(description)


if __name__ == "__main__":
    app_wrap_loguru()
