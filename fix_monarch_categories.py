#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "loguru",
#     "langchain",
#     "langchain-core",
#     "langchain-openai",
#     "icecream",
#     "openai",
#     "requests",
# ]
# ///

import asyncio
import csv
import io
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
- Convert the input data to the required 8-column Monarch format in this exact order: Date, Merchant, Category, Account, Original Statement, Notes, Amount, Tags
- Map input columns as follows:
  * Date → Date
  * Merchant → Merchant  
  * Category → Category
  * Account → Account
  * Original Statement → Original Statement
  * Notes → Notes
  * Amount → Amount
  * Tags → Tags

**CSV File Details:**
- The input CSV contains: Date, Merchant, Category, Account, Original Statement, Notes, Amount, Tags
- Output must be exactly 8 columns in the specified order: Date, Merchant, Category, Account, Original Statement, Notes, Amount, Tags
- Do NOT include a header row in your output (the header will be added separately)
- If the input contains a header row, ignore it and only process data rows

**Amount Format Requirements:**
- Monarch uses positive numbers for income and negative numbers for expenses
- +$100.00 = income, -$100.00 = expense
- Ensure amounts follow this convention

**CSV Escaping Requirements:**
- Fields containing commas, quotes, or newlines MUST be enclosed in double quotes
- Double quotes within fields must be escaped by doubling them (e.g., "He said \\"Hello\\"")
- Newlines within fields should be preserved within quoted fields
- Empty fields should be represented as empty strings (not quoted unless they contain special characters)

**Output Requirements:**
- Output exactly 8 columns per row in the specified order
- Follow proper CSV escaping rules as specified above
- The output should NOT contain ``` or ```csv
- Do NOT output a header row in your response
- There should be the same number of output lines as input data lines (excluding any input header)
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=content),
        ]
    )


def format_csv_output(llm_output: str) -> str:
    """
    Post-process LLM output to ensure proper CSV formatting
    """
    lines = llm_output.strip().split("\n")
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    for line in lines:
        if not line.strip():
            continue
        # Parse the line as CSV to handle any existing escaping
        try:
            reader = csv.reader([line])
            row = next(reader)
            # Re-write with proper escaping
            writer.writerow(row)
        except csv.Error:
            # If parsing fails, try to split by comma and clean up
            row = [field.strip().strip('"') for field in line.split(",")]
            writer.writerow(row)

    return output.getvalue().strip()


async def a_fix(path: str, chunk_size: int, lines_per_chunk: int):
    from langchain_openai.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatOpenAI(model="gpt-4o-mini")
    ic(llm)
    # llm = langchain_helper.get_model(openai=True)

    # Print the header row first
    print("Date,Merchant,Category,Account,Original Statement,Notes,Amount,Tags")

    with open(path, "r") as file:
        lines = file.readlines()
        # Skip the first line if it's a header
        if lines and lines[0].strip().lower().startswith(("date,", "date\t")):
            lines = lines[1:]
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
            # Format the output properly
            formatted_ret = format_csv_output(ret)
            results[index] = formatted_ret
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
                    f"Processing chunks {start_chunk}-{i + 1}/{total_chunks}, estimated remaining time: {estimated_remaining_time_minutes:.2f} minutes"
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
