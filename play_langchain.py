#!python3

import os
import asyncio
import openai
import json
from icecream import ic
import typer
import sys
import random
import psutil
from rich.console import Console
from rich.text import Text
import rich
import re
from typeguard import typechecked
import tiktoken
import time
from typing import List
import signal
import ast
from pydantic import BaseModel
import discord
import aiohttp
import datetime
from io import BytesIO
from asyncer import asyncify
from discord.ext import commands
from discord.ui import Button, View, Modal
from loguru import logger
from rich import print as rich_print
from langchain.prompts import PromptTemplate

console = Console()
app = typer.Typer()
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import BraveSearch


llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(temperature=0)


@logger.catch()
def app_wrap_loguru():
    app()


# Google search setup
# https://github.com/hwchase17/langchain/blob/d0c7f7c317ee595a421b19aa6d94672c96d7f42e/langchain/utilities/google_search.py#L9


@app.command()
def financial_agent(stock: str):
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools = load_tools(["bing-search"], llm=llm)
    # braveSearch = BraveSearch()
    # tools += [braveSearch]
    agent = initialize_agent(
        tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run(
        f"""
              What's the price outlook for : {stock}?, What are the top 3 reasons for the stock to go up?, What are the top 3 reasons for the stock to go down?

The output should be of the form:

Stock: XXX
Price: XXX
Price 1 year ago: XXX
Price 1 year ahead: XXX
Price goes up because:
- Point 1
- Point 2
- Point 3
Price goes down because:
- Point 1
- Point 2
- Point 3
"""
    )


@app.command()
def product_recommendation(product: str):

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant"
    )

    human_message_template = (
        "Make a list of 10  good name for a company that makes {product}?"
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_message_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(product=product, snow="Not Used")  # Yukky,
    print(f"A company that makes {product} is called: \n{response}")


if __name__ == "__main__":
    app_wrap_loguru()
