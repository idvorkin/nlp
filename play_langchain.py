#!python3


import os
from pathlib import Path
import pickle
import sys
import pudb
import typer
from icecream import ic
import langchain_helper
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Replicate


from loguru import logger
from rich import print
from rich.console import Console

console = Console()
app = typer.Typer(no_args_is_help=True)


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


@logger.catch()
def app_wrap_loguru():
    app()


# Google search setup
# https://github.com/hwchase17/langchain/blob/d0c7f7c317ee595a421b19aa6d94672c96d7f42e/langchain/utilities/google_search.py#L9


@app.command()
def tell_me_a_joke(count=4):
    chat = langchain_helper.get_model(google=True)
    template = ChatPromptTemplate.from_template(
        "Generate a list of exactly {count} joke about software engineers"
    )
    response = (template | chat).invoke({"count": count})
    ic(response)


def load_cached_prompt(prompt_name):
    from langchain import hub

    prompt_cache = os.path.expanduser("~/tmp/pickle/prompts")
    # if prompt_cache directory doesn't exist, create it
    if not os.path.exists(prompt_cache):
        os.makedirs(prompt_cache)
    prompt_maker_filename = f"{prompt_name.replace('/','_')}.pickle"
    prompt_maker_path = os.path.join(prompt_cache, prompt_maker_filename)

    if not os.path.exists(prompt_maker_path):
        prompt_maker_template = hub.pull(prompt_name)
        with open(prompt_maker_path, "wb") as f:
            pickle.dump(prompt_maker_template, f)
    else:
        with open(prompt_maker_path, "rb") as f:
            prompt_maker_template = pickle.load(f)

    return prompt_maker_template


@app.command()
def great_prompt(prompt):
    prompt_maker_template = load_cached_prompt("hardkothari/prompt-maker")
    model = langchain_helper.get_model(openai=True)
    chain = prompt_maker_template | model
    result = chain.invoke({"lazy_prompt": prompt, "task": prompt})
    print(result.content)


@app.command()
def summarize():
    prompt_maker_template = load_cached_prompt("langchain-ai/chain-of-density:ba34ae10")
    user_text = "".join(sys.stdin.readlines())
    model = langchain_helper.get_model(openai=True)
    chain = prompt_maker_template | model
    result = chain.invoke({"ARTICLE": user_text})
    ic(result)


@app.command()
def replicate():
    template = ChatPromptTemplate.from_template(
        "Generate a list of exactly 3 joke about software engineers"
    )
    ic()
    import time

    start_time = time.time()
    model = Replicate(
        # model="meta/meta-llama-3-8b-instruct",
        model="meta/meta-llama-3.1-405b-instruct",
        model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
    )
    chain = template | model
    result = chain.invoke({})
    end_time = time.time()
    execution_time = end_time - start_time
    ic(result)
    ic(f"Model execution time: {execution_time:.2f} seconds")


@app.command()
def search(query):
    from langchain_community.chat_models import ChatPerplexity

    # user_text = "".join(sys.stdin.readlines())
    chat = ChatPerplexity(model="llama-3-sonar-small-32k-online")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You provide searches"), ("human", query)]
    )
    chain = prompt | chat
    response = chain.invoke({"input": query})
    print(response.content)


@app.command()
def cat_pdf(path: Path):
    ic(path)
    pass


@app.command()
def deepseek():
    """Process stdin through DeepSeek model and stream the output."""
    user_text = "".join(sys.stdin.readlines())

    model = langchain_helper.get_model(deepseek=True)

    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant."), ("human", "{input}")]
    )

    # Create chain with streaming
    chain = prompt | model

    # Stream the response
    response = chain.stream({"input": user_text})
    for chunk in response:
        print(chunk.content, end="", flush=True)
    print()  # Add final newline


@app.command()
def stdin(
    o4: bool = typer.Option(False, "--o4", help="Use O4-mini model"),
    deepseek: bool = typer.Option(False, "--deepseek", help="Use DeepSeek model"),
    claude: bool = typer.Option(False, "--claude", help="Use Claude model"),
    llama: bool = typer.Option(False, "--llama", help="Use Llama model"),
    google: bool = typer.Option(False, "--google", help="Use Google model"),
    gpt_4_mini: bool = typer.Option(False, "--gpt-4-mini", help="Use GPT-4-mini model"),
):
    """Process stdin through the selected model and stream the output."""
    user_text = "".join(sys.stdin.readlines())

    # Get the appropriate model based on flags
    model = langchain_helper.get_model(
        o4_mini=o4,
        deepseek=deepseek,
        claude=claude,
        llama=llama,
        google=google,
        gpt_4_mini=gpt_4_mini,
    )

    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant."), ("human", "{input}")]
    )

    # Create chain with streaming
    chain = prompt | model

    # Stream the response
    response = chain.stream({"input": user_text})
    for chunk in response:
        print(chunk.content, end="", flush=True)
    print()  # Add final newline


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
