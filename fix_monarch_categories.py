#!python3

import asyncio
import time
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

Valid categories = "Groceries;Electronics;Pets;Entertainment & Recreation;Clothing;Furniture & Housewares;Shopping;Books;Movies;Fitness"

The output should be a copy of the input, replacing unknown with the category name, and nothing else, ther should not be any ``` quotes

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )


async def a_fix(path: str, chunk_size: int, lines_per_chunk: int):
    from langchain_openai.chat_models import ChatOpenAI

    # llm = ChatOpenAI(model="gpt-4o-mini")
    llm = ChatOpenAI(model="gpt-4o")
    ic(llm)
    # llm = langchain_helper.get_model(openai=True)
    with open(path, "r") as file:
        lines = file.readlines()
        total_chunks = (len(lines) + lines_per_chunk - 1) // lines_per_chunk  # Calculate total chunks
        start_time = time.time()
        async def process_chunk(chunk, index):
            ic(openai_wrapper.num_tokens_from_string(chunk))
            ret = (prompt_fix_categories(chunk) | llm | StrOutputParser()).invoke({})
            print(f"Chunk {index + 1} result: {ret}")

        tasks = []
        for i in range(total_chunks):
            chunk = "".join(lines[i * lines_per_chunk : (i + 1) * lines_per_chunk])
            tasks.append(process_chunk(chunk, i))
            if len(tasks) == chunk_size or i == total_chunks - 1:
                elapsed_time = time.time() - start_time
                average_time_per_chunk = elapsed_time / (i + 1)
                estimated_remaining_time = average_time_per_chunk * (total_chunks - (i + 1))
                estimated_remaining_time_minutes = estimated_remaining_time / 60
                ic(
                    f"Processing chunks {i-8}-{i+1}/{total_chunks}, estimated remaining time: {estimated_remaining_time_minutes:.2f} minutes"
                )
                await asyncio.gather(*tasks)
                tasks = []


app = typer.Typer(no_args_is_help=True)


@app.command()
def fix(
    trace: bool = False,
    path: str = typer.Argument(None),
    chunk_size: int = typer.Option(10, help="Number of chunks to process concurrently"),
    lines_per_chunk: int = typer.Option(10, help="Number of lines per chunk"),
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_fix(path, chunk_size, lines_per_chunk))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
