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

llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(temperature=0)


@logger.catch()
def app_wrap_loguru():
    app()


@app.command()
def product_recommendation(product: str):

    template = "You are a helpful assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Make a list of 10  good name for a company that makes {product}?"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(product=product, snow="Not Used")  # Yukky,
    print(f"A company that makes {product} is called: \n{response}")


if __name__ == "__main__":
    app_wrap_loguru()
