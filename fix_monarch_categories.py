#!python3

import asyncio
import typer
from loguru import logger
from rich import print
from rich.console import Console
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core import messages
import langchain_helper

def prompt_fix_categories(content):
    instructions = """
Please help me fix the categories in the monarch CSV input. Change any 'unknown' categories to the correct category based on the context provided.
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )

async def a_fix(path: str):
    llm = langchain_helper.get_model(claude=True)
    user_text = langchain_helper.get_text_from_path_or_stdin(path)
    ret = (prompt_fix_categories(user_text) | llm | StrOutputParser()).invoke({})
    print(ret)

console = Console()
app = typer.Typer(no_args_is_help=True)

@app.command()
def fix(
    trace: bool = False,
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_fix(path))
    )

@logger.catch()
def app_wrap_loguru():
    app()

if __name__ == "__main__":
    app_wrap_loguru()
