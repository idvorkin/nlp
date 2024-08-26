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
You are an expert in enhancing CSV data. You will be provided with segments of a CSV file, which will later be combined into a complete dataset. Your main tasks are to accurately categorize the data and ensure proper formatting for seamless parsing.

**Task Instructions:**
- Replace any 'Uncategorized' entries with the most suitable category based on the context provided.
- Valid categories include: "Groceries; Electronics; Pets; Entertainment & Recreation; Clothing; Furniture & Housewares; Shopping; Books; Movies; Fitness".
- Ensure all fields are correctly quoted to handle commas and quotes within the data. Each row must have the correct number of columns.

**CSV File Details:**
- The input CSV contains the following columns: Date, Merchant, Category, Account, Original Statement, Notes, Amount, Tags.

**Output Requirements:**
- Only output the column names and data present in the original input segment.
- Ensure your output is a valid CSV with the same number of columns as the input.
- Retain all lines from the input, except for modifying 'Uncategorized' entries to the correct category name.
- The output should NOT contain ``` or ```csv
- There should be the same number of output lines as input lines
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )


async def a_fix(path: str, chunk_size: int, lines_per_chunk: int):
    from langchain_openai.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatOpenAI(model="gpt-4o-mini")
    ic(llm)
    # llm = langchain_helper.get_model(openai=True)
    with open(path, "r") as file:
        lines = file.readlines()
        total_chunks = (
            len(lines) + lines_per_chunk - 1
        ) // lines_per_chunk  # Calculate total chunks
        start_time = time.time()
        results = [None] * total_chunks

        async def process_chunk(chunk, index):
            start_chunk_time = time.time()
            ic(openai_wrapper.num_tokens_from_string(chunk))
            ret = await (
                prompt_fix_categories(chunk) | llm | StrOutputParser()
            ).ainvoke({})
            results[index] = ret
            end_chunk_time = time.time()
            chunk_processing_time = end_chunk_time - start_chunk_time
            ic(f"Chunk {index + 1} processed in {chunk_processing_time:.2f} seconds")

        tasks = []
        for i in range(total_chunks):
            chunk = "".join(lines[i * lines_per_chunk : (i + 1) * lines_per_chunk])
            tasks.append(process_chunk(chunk, i))
            if len(tasks) == chunk_size or i == total_chunks - 1:
                elapsed_time = time.time() - start_time
                average_time_per_chunk = elapsed_time / (i + 1)
                estimated_remaining_time = average_time_per_chunk * (
                    total_chunks - (i + 1)
                )
                estimated_remaining_time_minutes = estimated_remaining_time / 60
                start_chunk = max(0, i - chunk_size + 1)
                ic(
                    f"Processing chunks {start_chunk}-{i+1}/{total_chunks}, estimated remaining time: {estimated_remaining_time_minutes:.2f} minutes"
                )
                await asyncio.gather(*tasks)
                tasks = []

        for i, result in enumerate(results):
            print(result)


app = typer.Typer(no_args_is_help=True)


@app.command()
def fix(
    trace: bool = False,
    path: str = typer.Argument(None),
    chunk_size: int = typer.Option(
        200, help="Number of chunks to process concurrently"
    ),
    lines_per_chunk: int = typer.Option(40, help="Number of lines per chunk"),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_fix(path, chunk_size, lines_per_chunk))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
