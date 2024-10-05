#!python3

import os

import typer
from icecream import ic
from loguru import logger
from rich import print
from rich.console import Console
import requests
import ell
from ell import Message
import openai_wrapper

ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")
ell.init(store=ELL_LOGDIR, autocommit=True)


DEFAULT_SEARCH_QUERY = "What's the weather in moscow"

console = Console()
app = typer.Typer(no_args_is_help=True)


@ell.tool()
def journal_append(content: str):
    """Append content to the journal"""
    return call_tony_server_as_vapi("journal-append", content=content)


@ell.tool()
def journal_read(date: str):
    """Read  the journal"""
    return call_tony_server_as_vapi("journal-read", date=date)


@ell.tool()
def library_arrivals():
    """When the bus gets to the library, which is the bus stop for garfield, when user asks when is the next bus to garfield"""
    return call_tony_server_as_vapi("library-arrivals")


@ell.tool()
def search(question: str):
    """Search the web"""
    return call_tony_server_as_vapi("search", question=question)


def call_tony_server_as_vapi(api, **kwargs):
    """Call the Tony server as it would be called by VAPI"""

    auth_headers = {"x-vapi-secret": os.getenv("TONY_API_KEY")}
    # url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    response = requests.post(url, json=kwargs, headers=auth_headers).json()
    return str(response)


TONY_TOOLS = [journal_append, journal_read, search, library_arrivals]


@ell.complex(model=openai_wrapper.get_ell_model(openai=True), tools=TONY_TOOLS)
def prompt_to_llm(message_history: list[Message]):
    return (
        [
            # the first message will be the system message
        ]
        + message_history
    )


def get_tony_server_url(dev_server: bool) -> str:
    """Select the appropriate server URL based on the dev_server flag."""
    if dev_server:
        return "https://idvorkin--modal-tony-server-assistant-dev.modal.run"
    else:
        return "https://idvorkin--modal-tony-server-assistant.modal.run"


# @ell.complex(model="gpt-4o-
# def call_tony()


@app.command()
def tony(dev_server: bool = False):
    """Talk to Toni"""

    ic("v0.0.4")

    ic("++assistant.api")
    payload = {"ignored": "ignored"}
    url_tony = get_tony_server_url(dev_server)
    ic(url_tony)
    assistant_response = requests.post(url_tony, json=payload)
    if assistant_response.status_code != 200:
        ic(assistant_response)
        return

    model_from_assistant = assistant_response.json()["assistant"]["model"]["model"]
    ic(model_from_assistant)
    ic("--assistant.api")

    messages = []

    # TODO build a program to parse this out
    system_message_content = assistant_response.json()["assistant"]["model"][
        "messages"
    ][0]["content"]
    messages.append(ell.system(system_message_content))

    while True:
        # if there's a tool response, we need to call the model again
        user_input = input("Igor:")
        if user_input == "debug":
            ic(model_from_assistant)
            continue
        if user_input == "search":
            ic("hardcode test")
            user_input = DEFAULT_SEARCH_QUERY
        messages.append(ell.user(user_input))
        # ic(custom_instructions)

        tony_response = prompt_to_llm(messages)  # type: ignore
        if tony_response.tool_calls:
            messages.append(tony_response)
            next_message = tony_response.call_tools_and_collect_as_message()
            messages.append(next_message)
            # call tony again
            tony_response = prompt_to_llm(messages)

        messages.append(tony_response)

        print(f"[yellow]Tony:{tony_response.text}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
