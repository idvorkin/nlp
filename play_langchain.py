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
from rich import print
import re
from typeguard import typechecked
import tiktoken
import time
from typing import List, Callable
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
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)


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


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        # NOTE: Not sure why doing it merged?
        # I guess this saves tokens (??)
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message


@app.command()
def dnd(protagonist_name="Donald Trump", quest="Find all the social security spending"):
    storyteller_name = "Dungeon Master"
    ic(quest)
    ic(protagonist_name)

    word_limit = 50  # word limit for task brainstorming

    game_description = f"""Here is the topic for a Dungeons & Dragons game: {quest}.
    There is one player in this game: the protagonist, {protagonist_name}.
    The story is narrated by the storyteller, {storyteller_name}."""

    player_descriptor_system_message = SystemMessage(
        content="You can add detail to the description of a Dungeons & Dragons player."
    )

    protagonist_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            Please reply with a creative description of the protagonist, {protagonist_name}, in {word_limit} words or less.
            Speak directly to {protagonist_name}.
            Do not add anything else."""
        ),
    ]

    protagonist_description = ChatOpenAI(temperature=1.0)(
        protagonist_specifier_prompt
    ).content

    storyteller_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""{game_description}
            Please reply with a creative description of the storyteller, {storyteller_name}, in {word_limit} words or less.
            Speak directly to {storyteller_name}.
            Do not add anything else."""
        ),
    ]

    storyteller_description = ChatOpenAI(temperature=1.0)(
        storyteller_specifier_prompt
    ).content

    # dump
    ic(protagonist_description)
    ic(storyteller_description)

    protagonist_system_message = SystemMessage(
        content=(
            f"""{game_description}
                Never forget you are the protagonist, {protagonist_name}, and I am the storyteller, {storyteller_name}.
                Your character description is as follows: {protagonist_description}.
                You will propose actions you plan to take and I will explain what happens when you take those actions.
                Speak in the first person from the perspective of {protagonist_name}.
                For describing your own body movements, wrap your description in '*'.
                Do not change roles!
                Do not speak from the perspective of {storyteller_name}.
                Do not forget to finish speaking by saying, 'It is your turn, {storyteller_name}.'
                Do not add anything else.
                Remember you are the protagonist, {protagonist_name}.
                Stop speaking the moment you finish speaking from your perspective.
                """
        )
    )

    storyteller_system_message = SystemMessage(
        content=(
            f"""{game_description}
            Never forget you are the storyteller, {storyteller_name}, and I am the protagonist, {protagonist_name}.
            Your character description is as follows: {storyteller_description}.
            I will propose actions I plan to take and you will explain what happens when I take those actions.
            Speak in the first person from the perspective of {storyteller_name}.
            For describing your own body movements, wrap your description in '*'.
            Do not change roles!
            Do not speak from the perspective of {protagonist_name}.
            Do not forget to finish speaking by saying, 'It is your turn, {protagonist_name}.'
            Do not add anything else.
            Remember you are the storyteller, {storyteller_name}.
            Stop speaking the moment you finish speaking from your perspective.
            """
        )
    )

    quest_specifier_prompt = [
        SystemMessage(content="You can make a task more specific."),
        HumanMessage(
            content=f"""{game_description}

            You are the storyteller, {storyteller_name}.
            Please make the quest more specific. Be creative and imaginative.
            Please reply with the specified quest in {word_limit} words or less.
            Speak directly to the protagonist {protagonist_name}.
            Do not add anything else."""
        ),
    ]
    specified_quest = ChatOpenAI(temperature=1.0)(quest_specifier_prompt).content
    ic(f"Original quest:\n{quest}\n")
    ic(f"Detailed quest:\n{specified_quest}\n")

    protagonist = DialogueAgent(
        name=protagonist_name,
        system_message=protagonist_system_message,
        model=ChatOpenAI(temperature=0.2),
    )

    storyteller = DialogueAgent(
        name=storyteller_name,
        system_message=storyteller_system_message,
        model=ChatOpenAI(temperature=0.2),
    )

    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        idx = step % len(agents)
        return idx

    max_iters = 6
    n = 0

    simulator = DialogueSimulator(
        agents=[storyteller, protagonist], selection_function=select_next_speaker
    )
    simulator.reset()
    simulator.inject(storyteller_name, specified_quest)
    print(f"[yellow]{storyteller_name}[/yellow]: {specified_quest}")
    print("\n")

    while n < max_iters:
        name, message = simulator.step()
        print(f"[cyan]{name}[/cyan]: {message}")
        print("\n")
        n += 1


@app.command()
def docs():
    from langchain.document_loaders import DirectoryLoader

    loader = DirectoryLoader(os.path.expanduser("~/blog"), glob="**/*.md")
    docs = loader.load()
    ic(len(docs))
    ic(docs[34])
    ic(docs[25])
    ic(docs[1])


if __name__ == "__main__":
    app_wrap_loguru()
