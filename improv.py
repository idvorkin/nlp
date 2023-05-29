#!python3

import os
import openai
import json
from icecream import ic
import typer
import sys
from rich import print as rich_print
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
    return num_tokens


def ask_gpt(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=True,
    debug=False,
):
    text_model_best, tokens = process_u4(u4)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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
    responses = 1
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
    assert len(response_contents) == 1
    return response_contents[0]


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


class Fragment:
    def __init__(self, player, text, reasoning=""):
        self.player = player
        self.text = text
        self.reasoning = reasoning

    def __str__(self):
        if self.reasoning:
            return f'Fragment("{self.player}", "{self.text}", "{self.reasoning}")'
        else:
            return f'Fragment("{self.player}", "{self.text}")'

    def __repr__(self):
        return str(self)


def print_story(story: List[Fragment], show_story: bool):
    # Split on '.', but only if there isn't a list
    coach_color = "bold blue"
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
        console.line()

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
    Fragment("coach", "Once upon a time", "A normal story start"),
    Fragment("student", "there lived "),
    Fragment("coach", "a shrew named", "using shrew to make it intereting"),
    Fragment("student", "Sarah. Every day the shrew"),
]
example_1_out = example_1_in + [
    Fragment("coach", "smelled something that reminded her ", "give user a good offer")
]

example_2_in = [
    Fragment(
        "coach", "Once Upon a Time within ", "A normal story start, with a narrowing"
    ),
    Fragment("student", "there lived a donkey"),
    Fragment("coach", "who liked to eat", "add some color"),
    Fragment("student", "Brocolli. Every"),
]

example_2_out = example_2_in + [
    Fragment("coach", "day the donkey", "continue in the format"),
]


def prompt_gpt_to_append_fragment_written_by_coach(story_so_far: List[Fragment]):
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

Now, here is the story we're doing together. Add the next coach fragment to the story

--
Actual Input:

{story_so_far}

Ouptut:
"""


def safe_eval_of_fragment_list(fragment_list_as_text: str) -> List[Fragment]:

    # input is valid python to construct fragments from text...
    # [ Fragment(..),Fragment(..)]
    # we don't want to run eval on that as it can run arbitrary stuff
    # we cna remove the word fragment to make it a list of tuples
    # then rebuild the tuples into fragments

    fragment_list_as_text = fragment_list_as_text.replace("Fragment", "")
    fragment_list = ast.literal_eval(fragment_list_as_text)
    fragments = [Fragment(*tuple) for tuple in fragment_list]
    return fragments


def get_user_input():
    console.print("[yellow] >>[/yellow]", end="")
    return input()


@app.command()
def code(
    debug: bool = typer.Option(False),
    u4: bool = typer.Option(False),
):
    """
    Play improv with GPT, prompt it to extend the story, but story is passed as List[Fragment]; Fragment=[Player,Text,Reasoning]
    """

    default_story_start = [
        Fragment("coach", "Once upon a time", "A normal story start"),
    ]
    story = default_story_start

    while True:
        print_story(story, show_story=True)

        user_says = get_user_input()
        story += [Fragment("student", user_says)]

        prompt = prompt_gpt_to_append_fragment_written_by_coach(story)

        valid_python_code_for_list_of_fragments = ask_gpt(
            prompt_to_gpt=prompt,
            debug=debug,
            u4=u4,
        )
        story = safe_eval_of_fragment_list(valid_python_code_for_list_of_fragments)
        ic(valid_python_code_for_list_of_fragments)


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
