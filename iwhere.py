#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "click==8.1.8", # Pinned click version
#     "icecream",
#     "rich",
#     "langchain",
#     "langchain-core",
#     "langchain-community",
#     "langchain-openai",
#     "langchain-groq",
#     "faiss-cpu",
#     "openai",
#     # "fastapi", # FastAPI import removed as the local server instance is removed.
#     "requests",
#     "typing-extensions",
#     "tiktoken",
#     "loguru",
#     "tqdm",
# ]
# ///

#!python3
import asyncio

# from langchain.callbacks.tracers.langchain import LangChainTracer # Moved to askig_logic
# import requests # Moved to askig_logic (for build_markdown_to_url_map)
# from functools import lru_cache # Moved to askig_logic
# import pathlib # No longer directly used by iwhere.py
from rich.console import Console
from icecream import ic  # Used by chunk_md, and potentially for CLI debugging
import typer
import os
from rich import print  # Used by CLI part

# from typing import List, Optional # No longer directly needed at top level of iwhere.py
# from langchain.docstore.document import Document # Moved to askig_build_logic
# from langchain_community.vectorstores import FAISS  # Moved to askig_build_logic
# import langchain_helper # Not directly used by iwhere.py CLI functions after refactor
# from langchain import (
#     text_splitter, # Moved to askig_build_logic
# )
from typing_extensions import Annotated

# from fastapi import FastAPI # Removed as server instance is no longer here
from openai_wrapper import setup_gpt  # num_tokens_from_string moved to build_logic

# from pathlib import Path # Moved to askig_build_logic
from loguru import logger  # Used by app_wrap_loguru and CLI command logging
import sys
import importlib.metadata

# Import from new local modules
import askig_logic  # For calling the core iask and iask_where logic
import askig_build_logic  # For calling the build logic

# --- Constants like EXCLUDED_DIRS and chunk_size_tokens moved to askig_build_logic.py ---

# --- Global gpt_model and FastAPI server (CLI specific or other uses) ---
gpt_model = setup_gpt()  # This might be for other CLI tools or legacy.
# server = FastAPI() # Removed: This local FastAPI server instance was unused after refactor.

app = typer.Typer(no_args_is_help=True)
console = Console()

# --- Build-related functions moved to askig_build_logic.py ---
# def should_exclude_path(path_str): ...
# def get_blog_content(path="~/blog"): ...
# def chunk_documents_recursive(documents, chunk_size=chunk_size_tokens): ...
# def chunk_documents_as_md(documents, chunk_size=chunk_size_tokens): ...
# def chunk_documents_as_md_large(documents, chunk_size=chunk_size_tokens): ...
# def dedup_chunks(chunks): ...
# def process_chunks_in_batches(chunks, batch_size=50): ...


@app.command()
def build(
    blog_path: Annotated[
        str, typer.Option(help="Path to the blog repository")
    ] = "~/blog",
    batch_size: Annotated[
        int,
        typer.Option(help="Number of chunks to process in each batch for embedding"),
    ] = 100,
):
    """Build the vector database from blog content."""
    logger.info(
        f"CLI 'build' invoked for blog path: {blog_path} with batch size: {batch_size}"
    )
    try:
        # The DB persist directory is now sourced from askig_build_logic.DEFAULT_FAISS_DB_DIR
        # or could be made a configurable option here too if desired.
        # For now, using the one defined in askig_build_logic for consistency during build.
        db_target_directory = askig_build_logic.DEFAULT_FAISS_DB_DIR

        askig_build_logic.perform_build(
            blog_repo_path=blog_path,
            db_persist_directory=db_target_directory,
            batch_size_for_embedding=batch_size,
        )
        print(f"✅ Blog database successfully built and saved to {db_target_directory}")
    except Exception as e:
        logger.error(f"CLI 'build' command failed: {e}")
        logger.exception("Traceback for build failure in CLI:")
        print(f"❌ Error during database building: {e}. Check logs for details.")
        raise typer.Exit(code=1)  # Indicate failure to the shell


@app.command()
def chunk_md(
    path: Annotated[str, typer.Argument()] = "~/blog/_posts/2020-04-01-Igor-Eulogy.md",
):
    """(Standalone) Partition a single markdown file using Unstructured. Requires 'unstructured' library."""
    # This command might need 'unstructured' to be added to dependencies if not already covered
    # For now, it remains as a utility, independent of the main build pipeline.
    try:
        from unstructured.partition.md import partition_md

        expanded_path = os.path.expanduser(path)
        logger.info(f"Partitioning markdown file: {expanded_path}")
        elements = partition_md(filename=expanded_path)
        ic(elements)
    except ImportError:
        logger.error(
            "'unstructured' library not found. Please install it to use chunk_md. Try: pip install unstructured"
        )
        print(
            "Error: 'unstructured' library not found. Please install it. Try: pip install unstructured"
        )
    except Exception as e:
        logger.error(f"Error during chunk_md for {path}: {e}")
        print(f"Error processing {path}: {e}")


def select_model_for_cli(model: str | None, fast: bool) -> str:
    """DRY helper to select the model name for CLI commands."""
    if model:
        return model
    return "llama" if fast else "openai"


# --- Updated Typer commands to use askig_logic --- (These remain as they were in the previous step)
@app.command()
def ask(
    question: Annotated[
        str, typer.Argument()
    ] = "What are the roles from Igor's Eulogy, answer in bullet form",
    facts: Annotated[
        int, typer.Option(help="Number of documents to use for context")
    ] = 20,
    debug: bool = typer.Option(False),
    model: Annotated[
        str,
        typer.Option(
            help="Model to use: openai, claude, llama, google, etc. (default: llama if --fast, openai if --no-fast)"
        ),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            help="Use a fast local model (llama). Default: True", is_flag=True
        ),
    ] = True,
):
    """Ask a question about Igor's blog content."""
    if debug:
        logger.info(
            f"CLI 'ask' called with: question='{question}', facts={facts}, model='{model}', debug={debug}, fast={fast}"
        )
    model_to_use = select_model_for_cli(model, fast)
    try:
        response = asyncio.run(
            askig_logic.iask_logic(question, facts, debug, model_to_use)
        )
        print(response)
    except Exception as e:
        logger.error(f"CLI 'ask' command failed: {e}")
        logger.exception("Traceback for ask failure in CLI:")
        print(f"❌ Error during ask: {e}. Check logs.")
        raise typer.Exit(code=1)


@app.command()
def where(
    topic: Annotated[str, typer.Argument(help="Topic to find placement for")],
    num_docs: Annotated[
        int, typer.Option(help="Number of documents to use for context")
    ] = 20,
    debug: Annotated[bool, typer.Option(help="Show debugging information")] = False,
    model: Annotated[
        str,
        typer.Option(
            help="Model to use: openai, claude, llama, google, etc. (default: llama if --fast, openai if --no-fast)"
        ),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            help="Use a fast local model (llama). Default: True", is_flag=True
        ),
    ] = True,
):
    """Suggest where to add new blog content about a topic."""
    if debug:
        logger.info(
            f"CLI 'where' called with: topic='{topic}', num_docs={num_docs}, model='{model}', debug={debug}, fast={fast}"
        )
    model_to_use = select_model_for_cli(model, fast)
    try:
        response = asyncio.run(
            askig_logic.iask_where_logic(topic, num_docs, debug, model_to_use)
        )
        print(response)
    except Exception as e:
        logger.error(f"CLI 'where' command failed: {e}")
        logger.exception("Traceback for where failure in CLI:")
        print(f"❌ Error during where: {e}. Check logs.")
        raise typer.Exit(code=1)


@app.command()
def versions():
    """Show versions of key libraries."""
    # Added askig_logic and askig_build_logic to demonstrate they are modules, though they don't have a standard version.
    libraries = [
        "typer",
        "click",
        "rich",
        "langchain",
        "fastmcp",
        "openai",
        "pydantic",
        "loguru",
        "uvicorn",
    ]  # Removed fastapi, added uvicorn as it's in askig_server
    print("Installed library versions:")
    for lib in libraries:
        try:
            version = importlib.metadata.version(lib)
            print(f"  {lib}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {lib}: Not found")
    print(
        "Custom modules: askig_logic, askig_models, askig_server, askig_build_logic (version not applicable)"
    )


def app_wrap_loguru():
    """Configure logging with loguru for console and file output for CLI."""
    log_level = "DEBUG" if os.getenv("ASKIG_DEBUG_LOGGING") else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    logger.info(f"iwhere.py CLI application started with log level: {log_level}")
    return app()


if __name__ == "__main__":
    # To enable debug logging for the CLI and potentially underlying logic files (if they use the same logger instance or respect root logger level):
    # export ASKIG_DEBUG_LOGGING=true
    app_wrap_loguru()
