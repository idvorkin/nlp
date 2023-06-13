#!python3

import os
import asyncio
import openai
import json
from icecream import ic
import typer
import sys
import random
from rich import print as rich_print
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
import fastapi
import uvicorn
from pydantic import BaseModel
import discord
import aiohttp
import datetime
from io import BytesIO
from asyncer import asyncify
from discord.ext import commands
from discord.ui import Button, View, Modal

# import OpenAI exceptiions
from openai.error import APIError, InvalidRequestError, AuthenticationError

# make a fastapi app called server
service = fastapi.FastAPI()

console = Console()

# By default, when you hit C-C in a pipe, the pipe is stopped
# with this, pipe continues
def keep_pipe_alive_on_control_c(sig, frame):
    sys.stdout.write(
        "\nInterrupted with Control+C, but I'm still writing to stdout...\n"
    )
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, keep_pipe_alive_on_control_c)

original_print = print
is_from_console = False

# text_model_best = "gpt-4"
text_model_best = "gpt-3.5-turbo"
code_model_best = "code-davinci-003"


# Load your API key from an environment variable or secret management service


def setup_gpt():
    PASSWORD = "replaced_from_secret_box"
    with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
        SECRETS = json.load(json_data)
        PASSWORD = SECRETS["openai"]
    openai.api_key = PASSWORD
    return openai


gpt3 = setup_gpt()
app = typer.Typer()


def num_tokens_from_string(string: str, encoding_name: str = "") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    num_tokens = num_tokens + 1  # for newline
    return num_tokens


def ask_gpt(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=True,
    debug=False,
):
    return ask_gpt_n(prompt_to_gpt, tokens=tokens, u4=u4, debug=debug, n=1)[0]


def ask_gpt_n(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=True,
    debug=False,
    n=1,
):
    text_model_best, tokens = process_u4(u4)
    messages = [
        {"role": "system", "content": "You are a really good improv coach."},
        {"role": "user", "content": prompt_to_gpt},
    ]

    input_tokens = num_tokens_from_string(prompt_to_gpt, "cl100k_base") + 100
    output_tokens = tokens - input_tokens

    if debug:
        ic(text_model_best)
        ic(tokens)
        ic(input_tokens)
        ic(output_tokens)

    start = time.time()
    responses = n
    response_contents = ["" for x in range(responses)]
    for chunk in openai.ChatCompletion.create(
        model=text_model_best,
        messages=messages,
        max_tokens=output_tokens,
        n=responses,
        temperature=0.7,
        stream=True,
    ):
        if not "choices" in chunk:
            continue

        for elem in chunk["choices"]:  # type: ignore

            delta = elem["delta"]
            delta_content = delta.get("content", "")
            response_contents[elem["index"]] += delta_content
    if debug:
        out = f"All chunks took: {int((time.time() - start)*1000)} ms"
        ic(out)

    # hard code to only return first response
    return response_contents


def process_u4(u4, tokens=0):
    is_token_count_the_default = tokens == 0  # TBD if we can do it without hardcoding.
    if u4:
        if is_token_count_the_default:
            tokens = 7800
        return "gpt-4", tokens
    else:
        if is_token_count_the_default:
            tokens = 3800
        return text_model_best, tokens


class Fragment(BaseModel):
    player: str
    text: str
    reasoning: str = ""

    def __str__(self):
        if self.reasoning:
            return f'Fragment("{self.player}", "{self.text}", "{self.reasoning}")'
        else:
            return f'Fragment("{self.player}", "{self.text}")'

    def __repr__(self):
        return str(self)

    # a static constructor that takes positional arguments
    @staticmethod
    def Pos(player, text, reasoning=""):
        return Fragment(player=player, text=text, reasoning=reasoning)


default_story_start = [
    Fragment.Pos("coach", "Once upon a time", "A normal story start"),
]


def print_story(story: List[Fragment], show_story: bool):
    # Split on '.', but only if there isn't a list
    coach_color = "bold bright_cyan"
    user_color = "bold yellow"

    def wrap_color(s, color):
        text = Text(s)
        text.stylize(color)
        return text

    def get_color_for(fragment):
        if fragment.player == "coach":
            return coach_color
        elif fragment.player == "student":
            return user_color
        else:
            return "white"

    console.clear()
    if show_story:
        console.print(story)
        console.rule()

    for fragment in story:
        s = fragment.text
        split_line = len(s.split(".")) == 2
        # assume it only contains 1, todo handle that
        if split_line:
            end_sentance, new_sentance = s.split(".")
            console.print(
                wrap_color(f" {end_sentance}.", get_color_for(fragment)), end=""
            )
            console.print(
                wrap_color(f"{new_sentance}", get_color_for(fragment)), end=""
            )
            continue

        console.print(wrap_color(f" {s}", get_color_for(fragment)), end="")

        # if (s.endswith(".")):
        #    rich_print(s)


example_1_in = [
    Fragment.Pos("coach", "Once upon a time", "A normal story start"),
    Fragment.Pos("student", "there lived "),
    Fragment.Pos("coach", "a shrew named", "using shrew to make it intereting"),
    Fragment.Pos("student", "Sarah. Every day the shrew"),
]
example_1_out = example_1_in + [
    Fragment.Pos(
        "coach", "smelled something that reminded her ", "give user a good offer"
    )
]

example_2_in = [
    Fragment.Pos(
        "coach", "Once Upon a Time within ", "A normal story start, with a narrowing"
    ),
    Fragment.Pos("student", "there lived a donkey"),
    Fragment.Pos("coach", "who liked to eat", "add some color"),
    Fragment.Pos("student", "Brocolli. Every"),
]

example_2_out = example_2_in + [
    Fragment.Pos("coach", "day the donkey", "continue in the format"),
]


def prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json(
    story_so_far: List[Fragment],
):
    # convert story to json
    story_so_far = json.dumps(story_so_far, default=lambda x: x.__dict__)
    return f"""
You are a professional improv performer and coach. Help me improve my improv skills through doing practice.
We're playing a game where we write a story together.
The story should have the following format
    - Once upon a time
    - Every day
    - But one day
    - Because of that
    - Because of that
    - Until finally
    - And ever since then

The story should be creative and funny

I'll write 1-5 words, and then you do the same, and we'll go back and forth writing the story.
The story is expressed as a json, I will pass in json, and you add the coach line to the json.
You will add a third field as to why you added those words in the line
Only add a single coach field to the output
You can correct spelling and capilization mistakes
The below strings are python strings, so if using ' quotes, ensure to escape them properly

Example 1 Input:

{example_1_in}

Example 1 Output:

{example_1_out}
--

Example 2 Input:

{example_2_in}

Example 2 Output:

{example_2_out}

--

Now, here is the story we're doing together. Add the next coach fragment to the story, and correct spelling and grammer mistakes in the fragments

--
Actual Input:

{story_so_far}

Ouptut:
"""


def get_user_input():
    console.print("[yellow] >>[/yellow]", end="")
    return input()


@app.command()
def json_objects(
    debug: bool = typer.Option(False),
    u4: bool = typer.Option(False),
):
    """
    Play improv with GPT, prompt it to extend the story, but story is passed back and forth as json
    """

    story = default_story_start

    while True:
        print_story(story, show_story=True)

        user_says = get_user_input()
        story += [Fragment(player="student", text=user_says)]

        prompt = (
            prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json(
                story
            )
        )

        json_version_of_a_story = ask_gpt(
            prompt_to_gpt=prompt,
            debug=debug,
            u4=u4,
        )

        # convert json_version_of_a_story to a list of fragments
        # Damn - Copilot wrote this code, and it's right (or so I think)
        story = json.loads(json_version_of_a_story, object_hook=lambda d: Fragment(**d))
        if debug:
            ic(json_version_of_a_story)
            ic(story)
            input("You can inspect, and then press enter to continue")


@app.command()
def server():
    # uvicorn.run("improv:server")
    import subprocess

    subprocess.run("uvicorn improv:service --reload")


# add a fastapi endpoint that takes the story so far as json, and returns the story with the coach fragment added
# you load the server on the command line by running uvicorn improv:server --reload
@service.get("/improv")
async def extend_story(story_so_far: List[Fragment]) -> List[Fragment]:
    # create the prompt
    prompt = prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json(
        story_so_far
    )

    # call gpt
    json_version_of_a_story = ask_gpt(
        prompt_to_gpt=prompt,
        debug=False,
        u4=False,
    )

    # convert json_version_of_a_story to a list of fragments
    # Damn - Copilot wrote this code, and it's right (or so I think)
    story = json.loads(json_version_of_a_story, object_hook=lambda d: Fragment(**d))

    return story


@app.command()
def text(
    debug: bool = typer.Option(False),
    u4: bool = typer.Option(True),
):
    """
    Play improv with GPT, prompt it to extend the story, where story is the text of the story so far.
    """
    prompt = """
You are a professional improv performer and coach. Help me improve my improv skills through doing practice.

We're playing a game where we write a story together.

The story should have the following format
    - Once upon a time
    - Every day
    - But one day
    - Because of that
    - Because of that
    - Until finally
    - And ever since then

The story should be creative and funny

I'll write 1-5 words, and then you do the same, and we'll go back and forth writing the story.
When you add words to the story, don't add more then 5 words, and stop in the middle of the sentance (that makes me be more creative)

The story we've written together so far is below and I wrote the last 1 to 5 words,
now add your words to the story (NEVER ADD MORE THEN 5 WORDS):
--


    """

    story = []  # (isCoach, word)

    while True:
        if debug:
            ic(prompt)

        coach_says = ask_gpt(prompt_to_gpt=prompt, debug=debug, u4=u4)
        story += [Fragment("coach", coach_says)]
        prompt += coach_says
        print_story(story, show_story=False)

        user_says = get_user_input()
        prompt += f" {user_says} "
        story += [Fragment("student", user_says)]


if __name__ == "__main__":
    app()
