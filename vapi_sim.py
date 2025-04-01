#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "icecream",
#     "loguru",
#     "rich",
#     "requests",
#     "ell-ai[sqlite]",
#     "alembic",
#     "groq",
#     "anthropic",
#     "openai",
#     "langchain-core",
#     "langchain-openai",
#     "pydantic",
# ]
# ///

import os
import typer
from icecream import ic
from loguru import logger
from rich import print
from rich.console import Console
import requests
import ell
from ell import Message
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option

DEFAULT_SEARCH_QUERY = "What's the weather in moscow"

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()

# Global flag to track if we're authenticated as Igor
is_authenticated_as_igor = False
# Global list to store all available tools
ALL_TOOLS = []


# Dynamic tool generation function
def create_tool_function(tool_name):
    """Create a dynamic tool function with the given name"""

    @ell.tool()
    def dynamic_tool(*args, **kwargs):
        """Dynamically created tool"""
        if is_authenticated_as_igor:
            return call_tony_server_as_igor(tool_name, **kwargs)
        else:
            return call_tony_server_as_vapi(tool_name, **kwargs)

    # Set the proper name and docstring
    dynamic_tool.__name__ = tool_name
    dynamic_tool.__doc__ = f"Dynamically created tool for {tool_name}"

    return dynamic_tool


# Basic tools we know exist
@ell.tool()
def journal_append(content: str):
    """Append content to the journal"""
    if is_authenticated_as_igor:
        return call_tony_server_as_igor("journal-append", content=content)
    else:
        return call_tony_server_as_vapi("journal-append", content=content)


@ell.tool()
def journal_read(date: str):
    """Read  the journal"""
    if is_authenticated_as_igor:
        return call_tony_server_as_igor("journal-read", date=date)
    else:
        return call_tony_server_as_vapi("journal-read", date=date)


@ell.tool()
def library_arrivals():
    """When the bus gets to the library, which is the bus stop for garfield, when user asks when is the next bus to garfield"""
    if is_authenticated_as_igor:
        return call_tony_server_as_igor("library-arrivals")
    else:
        return call_tony_server_as_vapi("library-arrivals")


@ell.tool()
def search(question: str):
    """Search the web"""
    if is_authenticated_as_igor:
        return call_tony_server_as_igor("search", question=question)
    else:
        return call_tony_server_as_vapi("search", question=question)


def call_tony_server_as_vapi(api, **kwargs):
    """Call the Tony server as it would be called by VAPI"""

    auth_headers = {"x-vapi-secret": os.getenv("TONY_API_KEY")}
    url = f"https://idvorkin--modal-tony-server-fastapi-app.modal.run/{api}"
    response = requests.post(url, json=kwargs, headers=auth_headers).json()
    return str(response)


def call_tony_server_as_igor(api, **kwargs):
    """Call the Tony server as Igor by adding his phone number"""

    auth_headers = {"x-vapi-secret": os.getenv("TONY_API_KEY")}
    # Add Igor's phone number to simulate a call from Igor
    if "message" not in kwargs:
        kwargs["message"] = {}

    # Add the phone call structure that would identify as Igor's number
    if "call" not in kwargs["message"]:
        kwargs["message"]["call"] = {}
    if "customer" not in kwargs["message"]["call"]:
        kwargs["message"]["call"]["customer"] = {}

    kwargs["message"]["call"]["customer"]["number"] = "+12068904339"  # Igor's number

    url = f"https://idvorkin--modal-tony-server-fastapi-app.modal.run/{api}"
    response = requests.post(url, json=kwargs, headers=auth_headers).json()
    return str(response)


TONY_TOOLS = [journal_append, journal_read, search, library_arrivals]


@ell.complex(model=get_ell_model(openai=True), tools=TONY_TOOLS)
def prompt_to_llm(message_history: list[Message]):
    """Default LLM prompt handler"""
    return (
        [
            # the first message will be the system message
        ]
        + message_history
    )


def get_tony_server_url(dev_server: bool) -> str:
    """Select the appropriate server URL based on the dev_server flag."""
    if dev_server:
        return "https://idvorkin--modal-tony-server-dev-fastapi-app.modal.run"
    else:
        return "https://idvorkin--modal-tony-server-fastapi-app.modal.run"


# @ell.complex(model="gpt-4o-
# def call_tony()


@app.command()
def tony(
    dev_server: bool = Option(False, help="Use the development server"),
    studio: bool = Option(False, help="Launch the ELL Studio interface"),
    port: int = Option(
        None, help="Port to run the ELL Studio on (only used with --studio)"
    ),
    call_as_igor: bool = Option(
        True, help="Authenticate as Igor to access restricted tools"
    ),
):
    """
    Talk to Tony or launch the ELL Studio interface.

    This command allows you to interact with Tony or open the ELL Studio for
    interactive model exploration and testing.
    """
    global is_authenticated_as_igor
    is_authenticated_as_igor = call_as_igor

    if studio:
        run_studio(port=port)
        return

    ic("v0.0.4")

    ic("++assistant.api")
    payload = {"message": {"type": "assistant-request"}}

    # Add Igor's phone number if requested
    if call_as_igor:
        ic("Authenticating as Igor")
        if "call" not in payload["message"]:
            payload["message"]["call"] = {}
        if "customer" not in payload["message"]["call"]:
            payload["message"]["call"]["customer"] = {}

        payload["message"]["call"]["customer"]["number"] = (
            "+12068904339"  # Igor's number
        )

    url_tony = get_tony_server_url(dev_server)
    ic(url_tony)
    assistant_response = requests.post(f"{url_tony}/assistant", json=payload)
    if assistant_response.status_code != 200:
        ic(assistant_response)
        return

    model_from_assistant = assistant_response.json()["assistant"]["model"]["model"]
    ic(model_from_assistant)

    # Print out the tools available from the server
    tools_from_assistant = assistant_response.json()["assistant"]["model"].get(
        "tools", []
    )
    ic("Available tools from server:")

    # Store available tool names for info and diagnostics
    available_tool_names = []

    for tool in tools_from_assistant:
        function_name = tool.get("function", {}).get("name", "unknown")
        available_tool_names.append(function_name)
        ic(f"Tool: {function_name}")

    # We're only going to use our predefined tools for now to keep it simple
    global ALL_TOOLS
    ALL_TOOLS = TONY_TOOLS
    ic(f"Using {len(ALL_TOOLS)} predefined tools")

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

        tony_response = prompt_to_llm(messages)  # Update to use all_tools
        if tony_response.tool_calls:
            messages.append(tony_response)
            next_message = tony_response.call_tools_and_collect_as_message()
            messages.append(next_message)
            # call tony again
            tony_response = prompt_to_llm(messages)  # Update here too

        messages.append(tony_response)

        print(f"[yellow]Tony:{tony_response.text}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
