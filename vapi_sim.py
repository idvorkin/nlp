#!python3

import os

import typer
from icecream import ic
from langchain_helper import get_model, get_model_name
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, ToolMessage
from loguru import logger
from rich import print
from rich.console import Console
import requests
from langchain_core.tools import StructuredTool, tool


DEFAULT_SEARCH_QUERY = "What's the weather in moscow"

console = Console()
app = typer.Typer(no_args_is_help=True)


@tool
def journal_append(content):
    """Append content to the journal"""
    return call_tony_server_as_vapi("journal-append", content=content)


@tool
def journal_read(date):
    """Read  the journal"""
    return call_tony_server_as_vapi("journal-read", date=date)


@tool
def library_arrivals():
    """When the bus gets to the library, which is the bus stop for garfield, when user asks when is the next bus to garfield"""
    return call_tony_server_as_vapi("library-arrivals")


@tool
def search(question):
    """Search the web"""
    return call_tony_server_as_vapi("search", question=question)


def call_tony_server_as_vapi(api, **kwargs):
    """Call the Tony server as it would be called by VAPI"""

    auth_headers = {"x-vapi-secret": os.getenv("TONY_API_KEY")}
    # url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    response = requests.post(url, json=kwargs, headers=auth_headers).json()
    return str(response)


def process_tool_calls(llm_result) -> list[ToolMessage]:
    TONY_TOOLS_STRUCTURED_HACK: list[StructuredTool] = TONY_TOOLS  # type:ignore - TONY_TOOLS are actually StructuredTools once wrapped
    return [
        process_tool_call(TONY_TOOLS_STRUCTURED_HACK, tool_call)
        for tool_call in llm_result.tool_calls
    ]


TONY_TOOLS = [journal_append, journal_read, search, library_arrivals]


# TODO: Consider transforming this into a nice lookup table
def process_tool_call(tools: list[StructuredTool], tool_call: dict):
    tool_name = tool_call["name"]
    get_name = lambda t: t.func.__name__ if t.func else "unknown"  # noqa - all functions should be valid

    # find the function in tony's too list
    tool = [t.func for t in tools if get_name(t) == tool_name][0]
    assert tool  # there needs to be a matching tool
    ic(tool_call["args"])
    tool_ret = tool(**tool_call["args"])
    return ToolMessage(tool_call_id=tool_call["id"], content=tool_ret)


def get_tony_server_url(dev_server: bool) -> str:
    """Select the appropriate server URL based on the dev_server flag."""
    if dev_server:
        return "https://idvorkin--modal-tony-server-assistant-dev.modal.run"
    else:
        return "https://idvorkin--modal-tony-server-assistant.modal.run"


@app.command()
def tony(dev_server: bool = False):
    """Talk to Toni"""

    # from langchain.chat_models import init_chat_model
    # model = init_chat_model(model_name).
    ic("v0.03")
    ic("++init model")
    model = get_model(openai=True).bind_tools(TONY_TOOLS)
    ic(get_model_name(model))  # type: ignore
    ic("--init model")

    ic("++assistant.api")
    payload = {"ignored": "ignored"}
    url_tony = get_tony_server_url(dev_server)
    ic(url_tony)
    tony_response = requests.post(url_tony, json=payload)
    if tony_response.status_code != 200:
        ic(tony_response)
        return

    model_from_assistant = tony_response.json()["assistant"]["model"]["model"]
    ic(model_from_assistant)
    ic("--assistant.api")

    memory = ChatMessageHistory()

    # TODO build a program to parse this out
    system_message_content = tony_response.json()["assistant"]["model"]["messages"][0][
        "content"
    ]
    memory.add_message(SystemMessage(content=system_message_content))

    while True:
        is_last_message_tool_response = isinstance(memory.messages[-1], ToolMessage)

        if not is_last_message_tool_response:
            # if there's a tool response, we need to call the model again
            user_input = input("Igor:")
            if user_input == "debug":
                ic(model_from_assistant)
                continue
            if user_input == "search":
                ic("hardcode test")
                user_input = DEFAULT_SEARCH_QUERY
            memory.add_user_message(message=user_input)
            # ic(custom_instructions)

        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        llm_result = chain.invoke({})  # type: ignore
        memory.add_message(llm_result)
        tool_responses = process_tool_calls(llm_result)
        if len(tool_responses):
            memory.add_messages(tool_responses)
            prompt = ChatPromptTemplate.from_messages(memory.messages)
            chain = prompt | model
            llm_result = chain.invoke({})  # type: ignore
            memory.add_message(llm_result)

        print(f"[yellow]Tony:{llm_result.content}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
