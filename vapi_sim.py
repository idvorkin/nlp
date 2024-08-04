#!python3

import os

import typer
from icecream import ic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from loguru import logger
from rich import print
from rich.console import Console
import requests
from langchain_core.tools import tool


console = Console()
app = typer.Typer()


@tool
def journal_append(content):
    """Append content to the journal"""
    return str(simulate_vapi_call("journal-append", content=content))


@tool
def journal_read(date):
    """Read  the journal"""
    return str(simulate_vapi_call("journal-read", date=date))


def auth_headers():
    return {"x-vapi-secret": os.getenv("TONY_API_KEY")}


def simulate_vapi_call(api, **kwargs):
    """Search the web"""
    # url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    url = f"https://idvorkin--modal-tony-server-{api}.modal.run"
    response = requests.post(url, json=kwargs, headers=auth_headers()).json()
    return str(response)


@tool
def search(question):
    """Search the web"""
    return simulate_vapi_call("search", question=question)


def process_tool_calls(llm_result):
    if len(llm_result.tool_calls) == 0:
        return None

    tool_calls = llm_result.tool_calls
    if len(tool_calls) > 1:
        assert len(tool_calls) == 1
        ic(tool_calls)

    tool_call = llm_result.tool_calls[0]
    ic(tool_call)
    tool_name = tool_call["name"]
    args = tool_call["args"]
    if tool_name == "search":
        search_result = search(args["question"])
        ic(search_result)
        return ToolMessage(tool_call_id=tool_call["id"], content=str(search_result))

    if tool_name == "journal_append":
        journal_append(args["content"])
        return ToolMessage(tool_call_id=tool_call["id"], content="")

    if tool_name == "journal_read":
        daily_log_content = journal_read(args["date"])
        return ToolMessage(tool_call_id=tool_call["id"], content=daily_log_content)

    ic("unsupported tool", tool_name)
    return ToolMessage(
        tool_call_id=tool_call["id"],
        content="tell user tool call failed",
    )


@app.command()
def tony():
    """Talk to Toni"""

    # from langchain.chat_models import init_chat_model
    # model = init_chat_model(model_name).
    ic("v0.02")
    ic("++init model")
    model = ChatOpenAI(model="gpt-4o").bind_tools(
        [journal_append, journal_read, search]
    )
    ic("--init model")

    ic("++assistant.api")
    payload = {"ignored": "ignored"}
    url_tony = "https://idvorkin--modal-tony-server-assistant.modal.run"
    ic(url_tony)
    tony_response = requests.post(url_tony, json=payload).json()
    model_name = tony_response["assistant"]["model"]["model"]
    ic(model_name)
    ic("--assistant.api")

    memory = ChatMessageHistory()

    # TODO build a program to parse this out
    system_message_content = tony_response["assistant"]["model"]["messages"][0][
        "content"
    ]
    memory.add_message(SystemMessage(content=system_message_content))

    while True:
        is_last_message_tool_response = isinstance(memory.messages[-1], ToolMessage)

        if not is_last_message_tool_response:
            # if there's a tool response, we need to call the model again
            user_input = input("Igor:")
            if user_input == "debug":
                ic(model_name)
                continue
            if user_input == "search":
                ic("hardcode test")
                user_input = "What's  the weather in moscow"
            memory.add_user_message(message=user_input)
            # ic(custom_instructions)

        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        llm_result: AIMessage = chain.invoke({})  # type: AIMessage
        memory.add_ai_message(llm_result)
        tool_respone = process_tool_calls(llm_result)
        if tool_respone:
            memory.add_message(tool_respone)
            prompt = ChatPromptTemplate.from_messages(memory.messages)
            chain = prompt | model
            llm_result: AIMessage = chain.invoke({})  # type: AIMessage
            memory.add_ai_message(llm_result)

        print(f"[yellow]Tony:{llm_result.content}")


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
