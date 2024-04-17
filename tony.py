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
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from dateutil import tz

console = Console()
app = typer.Typer()

TONY_ASSISTANT_ID = "f5fe3b31-0ff6-4395-bc08-bc8ebbbf48a6"


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
    End: datetime

    def length_in_seconds(self):
        return (self.End - self.Start).total_seconds()


def parse_call(call) -> Call:
    customer = ""
    if "customer" in call:
        customer = call["customer"]["number"]

    start = call.get("createdAt")
    # there is no end in some failure conditions
    end = call.get("endedAt", start)
    start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")
    end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S.%fZ")

    start_dt = start_dt.replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
    end_dt = end_dt.replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())

    return Call(
        Caller=customer,
        Transcript=call.get("transcript", ""),
        Start=start_dt,
        End=end_dt,
    )


def vapi_calls() -> list[Call]:
    # list all calls from VAPI
    # help:  https://api.vapi.ai/api#/Calls/CallController_findAll

    headers = {
        "authorization": f"{os.environ['VAPI_API_KEY']}",
        "createdAtGE": (datetime.now() - timedelta(days=1)).isoformat(),
    }
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
def update_tony():
    headers = {
        "authorization": f"{os.environ['VAPI_API_KEY']}",
    }
    tony = httpx.get(
        f"https://api.vapi.ai/assistant/{TONY_ASSISTANT_ID}", headers=headers
    ).json()
    # ic (tony)
    # original = tony["model"]["messages"][0]
    # print(original)
    # Run a diff to make sure happy with the path
    Path("tony.json").write_text(json.dumps(tony, indent=4))

    patch_document = [
        {"op": "replace", "path": "/model/messages", "value": tony["model"]["messages"]}
    ]

    ic(patch_document)

    patched = httpx.patch(
        f"https://api.vapi.ai/assistant/{TONY_ASSISTANT_ID}",
        headers=headers,
        json=patch_document,
    )
    ic(patched)
    ic(patched.text)
    # Ha, for now just copy the life convo to the clipboard, and paste it into the uX:
    # https://dashboard.vapi.ai/assistants


@app.command()
def parse_calls(
    trace: bool = False,
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_parse_calls())
    )


@app.command()
def local_debug():
    modal_storage = "modal_readonly"
    base = Path(f"{modal_storage}")
    assistant_txt = (base / "tony_assistant_spec.json").read_text(
        encoding="utf-8", errors="ignore"
    )
    ic(assistant_txt)
    tony = json.loads(assistant_txt)
    tony_prompt = json.loads((base / "tony_system_prompt.md").read_text())
    # update system prompt
    tony["assistant"]["model"]["messages"][0]["content"] = tony_prompt
    ic(len(tony))


async def a_parse_calls():
    async def transcribe_call(user_text):
        llm = langchain_helper.get_model(openai=True)
        callSummary: CallSummary = await (
            prompt_transcribe_call(user_text) | llm.with_structured_output(CallSummary)
        ).ainvoke({})  # type:ignore
        return callSummary

    calls = vapi_calls()

    def interesting_call(call: Call):
        return call.length_in_seconds() > 90 and "4339" in call.Caller

    interesting_calls = [call for call in calls if interesting_call(call)]
    for call in interesting_calls:
        start = call.Start.strftime("%Y-%m-%d %H:%M")
        ic(call.length_in_seconds(), call.Caller, start)
        call_summary = await transcribe_call(call.Transcript)
        print(call_summary)


if __name__ == "__main__":
    app_wrap_loguru()
