#!python3

import asyncio
import typer
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core import messages
import langchain_helper
from icecream import ic
import openai_wrapper

def prompt_fix_categories(content):
    instructions = """
Please help me fix the categories in the monarch CSV input. Change any 'Uncategorized' categories to the correct category based on the context provided.

Valid categories = "Groceries;Electronics;Pets;Entertainment & Recreation;Clothing;Furniture & Housewares;Shopping"

The output should be a copy of the input, replacing unknown with the category name, and nothing else, ther should not be any ``` quotes

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )

async def a_fix(path: str):
    from langchain_openai.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    ic(llm)
    #llm = langchain_helper.get_model(openai=True)
    with open(path, 'r') as file:
        lines = file.readlines()
        total_chunks = (len(lines) + 9) // 10  # Calculate total chunks
        for i in range(total_chunks):
            chunk = ''.join(lines[i*10:(i+1)*10])
            ic(f"Processing chunk {i+1}/{total_chunks}")
            ic(openai_wrapper.num_tokens_from_string(chunk))
            ret = (prompt_fix_categories(chunk) | llm | StrOutputParser()).invoke({})
            print(ret)

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
