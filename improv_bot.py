#!python3

import contextlib
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

    def __eq__(self, other):
        return (
            self.player == other.player
            and self.text == other.text
            and self.reasoning == other.reasoning
        )

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
Do not correct spelling and capilization mistakes
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


ic(discord)
bot = discord.Bot()

bot_help_text = "Replaced on_ready"


async def smart_send(ctx, message):
    is_message = not hasattr(ctx, "defer")
    return await ctx.channel.send(message) if is_message else await ctx.send(message)


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    global bot_help_text
    bot_help_text = f"""```

Commands:
 /once-upon-a-time - start a new story
 /continue  - continue the story
 /story  - print or continue the story
 /help - show this help
 /debug - show debug info
 /visualize - show a visualization of the story so far
 /explore - do a choose a your own adventure completion
When you DM the bot directly, or include a @{bot.user.display_name} in a channel
 - Add your own words to the story - extend the story
 - '.' - The bot will give you suggestions to continue
 - More coming ...
    ```"""


context_to_story = dict()


# Due to permissions, we should only get this for a direct message
@bot.event
async def on_message(message):
    ic(message)
    # await on_mention(message)


def key_for_ctx(ctx):
    is_dm_type = isinstance(ctx, discord.channel.DMChannel)
    is_dm_channel = is_dm_type or ctx.guild is None
    if is_dm_channel:
        return f"DM-{ctx.author.name}-{ctx.author.id}"
    else:
        return f"{ctx.guild.name}-{ctx.channel.name}"


def get_story_for_channel(ctx):
    key = key_for_ctx(ctx)
    if key not in context_to_story:
        reset_story_for_channel(ctx)

    # return a copy of the story
    return context_to_story[key][:]


def set_story_for_channel(ctx, story):
    key = key_for_ctx(ctx)
    # force a copy
    context_to_story[key] = story[:]


def reset_story_for_channel(ctx):
    set_story_for_channel(ctx, default_story_start)


# https://rebane2001.com/discord-colored-text-generator/
# We can colorize story for discord
def color_story_for_discord(story: List[Fragment]):
    def get_color_for(story, fragment: Fragment):
        if fragment.player == "coach":
            return ""
        else:
            return "**"

    def wrap_color(text, color):
        return f"{color}{text}{color}  "

    output = ""
    for fragment in story:
        s = fragment.text
        output += wrap_color(f"{s}", get_color_for(story, fragment))

    return "\n" + output


# Defines a custom button that contains the logic of the game.
# what the type of `self.view` is. It is not required.
class StoryButton(discord.ui.Button):
    def __init__(self, label, ctx, story):
        super().__init__(label=label, custom_id=f"button_{random.randint(0, 99999999)}")
        self.ctx = ctx
        self.story = story

    # This function is called whenever this particular button is pressed.
    async def callback(self, interaction: discord.Interaction):

        # My story has a new fragment appened, so should have equality on the previous line.
        latest_story = get_story_for_channel(self.ctx)
        active_story=self.story
        is_story_changed_under_us = latest_story[-1] != self.story[-2]
        if is_story_changed_under_us:
            colored = color_story_for_discord(latest_story)
            user_fragment = active_story[-1]
            response = f"Can't add **{user_fragment.text}** \nBecause the story has changed.\nThe current story is:\n{colored}"
            await interaction.response.edit_message(content=response, view=None)
        else:
            colored = color_story_for_discord(self.story)
            set_story_for_channel(self.ctx, self.story)
            await interaction.response.edit_message(content=colored, view=None)


@bot.command(description="Explore alternatives with the bot")
async def explore(ctx):
    async with make_slash_command_deferred(ctx):
        await explore_internal(ctx)

async def explore_internal(ctx):
    active_story = get_story_for_channel(ctx)

    colored = color_story_for_discord(active_story)
    # Can pass in a message or a context, silly pycord, luckily can cheat in pycord
    progress_message = await smart_send(ctx, ".")
    view = View()

    prompt = prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json(
        active_story
    )

    output_waiting_task = append_dots_to_message_task(progress_message, colored)

    n = 4
    list_of_json_version_of_a_story = await asyncify(ask_gpt_n)(
        prompt_to_gpt=prompt, debug=False, u4=False, n=n
    )
    output_waiting_task.cancel()

    # make stories from json
    list_of_stories = [
        json.loads(json_version_of_a_story, object_hook=lambda d: Fragment(**d))
        for json_version_of_a_story in list_of_json_version_of_a_story
    ]

    for story in list_of_stories:
        # add a button for the last fragment of each story
        view.add_item(StoryButton(label=story[-1].text[:70], ctx=ctx, story=story))

    await progress_message.edit(content=colored, view=view)

@bot.command(description="Start a new story with the bot")
async def once_upon_a_time(ctx, continue_with: discord.Option(str, name="continue_with", description="continue story with", required="False")
                           =""):
    reset_story_for_channel(ctx)
    active_story = get_story_for_channel(ctx)
    if continue_with:
        # extend what the coach said
        active_story[-1].text+=f" {continue_with}"
        set_story_for_channel(ctx, active_story)

    colored = color_story_for_discord(active_story)
    response = f"{bot_help_text}\n**The story so far:** {colored}"
    await ctx.respond(response)


@contextlib.asynccontextmanager
async def make_slash_command_deferred(ctx):
    is_slash_command = hasattr(ctx, "defer")
    try:
        if is_slash_command:
            await ctx.defer()
        yield
    finally:
        if is_slash_command:
            await ctx.followup.send(content="")

async def extend_story_for_bot(ctx, extend: str = ""):
    async with make_slash_command_deferred(ctx):
        await extend_story_for_bot_internal(ctx,extend)


async def extend_story_for_bot_internal(ctx, extend: str = ""):
    # if story is empty, then start with the default story
    active_story = get_story_for_channel(ctx)

    if not extend:
        # Called with an empty message lets send help as well
        colored = color_story_for_discord(active_story)
        await smart_send(ctx, f"{bot_help_text}\n**The story so far:** {colored}")
        return

    user_said = Fragment(player=ctx.author.name, text=extend)
    active_story += [user_said]

    ic(active_story)
    colored = color_story_for_discord(active_story)
    progress_message = await smart_send(ctx,".")
    output_waiting_task = append_dots_to_message_task(progress_message, colored)

    prompt = prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json(
        active_story
    )

    json_version_of_a_story = await asyncify(ask_gpt)(
        prompt_to_gpt=prompt,
        debug=False,
        u4=False,
    )
    output_waiting_task.cancel()

    ic(json_version_of_a_story)

    # convert json_version_of_a_story to a list of fragments
    # Damn - Copilot wrote this code, and it's right (or so I think)
    active_story = json.loads(
        json_version_of_a_story, object_hook=lambda d: Fragment(**d)
    )

    # The active story is the original story + user fragment + coach fragment, last original story is at -3
    #story changed if the last fragment is the one before we added the bot fragment
    latest_story = get_story_for_channel(ctx)
    is_story_changed_under_us = latest_story[-1] != active_story[-3]

    # It's possible someone edited the story before we got an answer from GPT.
    # When then happens, lets not update.
    if is_story_changed_under_us:
        colored = color_story_for_discord(latest_story)
        user_fragment = active_story[-2]
        response = f"Can't add {user_fragment.player} line of **{user_fragment.text}**\n Because the story has changed.\nThe current story is:\n{colored}"
        await progress_message.edit(content=response)
        return

    set_story_for_channel(ctx, active_story)

    # convert story to text
    print_story(active_story, show_story=True)
    colored = color_story_for_discord(active_story)

    await progress_message.edit(content=colored)


@bot.command(description="Show the story so far, or extend it")
async def story(
    ctx,
    extend: discord.Option(
        str, name="continue_with", description="continue story with", required="False"
    ),
):
    await extend_story_for_bot(ctx, extend)


@bot.command(name="continue", description="Continue the story")
async def extend(
    ctx, with_: discord.Option(str, name="with", description="continue story with")
):
    await extend_story_for_bot(ctx, with_)


@bot.command(description="Show help")
async def help(ctx):
    active_story = get_story_for_channel(ctx)
    colored = color_story_for_discord(active_story)
    response = f"{bot_help_text}\n**The story so far:** {colored}"
    await ctx.respond(response)


@bot.command(description="See local state")
async def debug(ctx):
    active_story = get_story_for_channel(ctx)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    debug_out = f"""```ansi
Process:
    Up time: {datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())}
    VM: {memory_info.vms / 1024 / 1024} MB
    Residitent: {memory_info.rss / 1024 / 1024} MB
    Shared: {memory_info.shared / 1024 / 1024} MB
Active Story:
    {[repr(f) for f in active_story] }
Other Stories
{context_to_story.keys()}
    ```
    """
    await ctx.respond(debug_out)


@app.command()
def run_bot():
    # read token from environment variable, or from the secret box, if in neither throw
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
            SECRETS = json.load(json_data)
            token = SECRETS["discord-improv-bot"]

    # throw if token not found
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
    bot.add_cog(MentionListener(bot))
    bot.run(token)


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@bot.command(description="Visualize the story so far")
async def visualize(ctx, count: int = 2):
    count = min(count, 8)
    active_story = get_story_for_channel(ctx)
    story_as_text = " ".join([f.text for f in active_story])
    await ctx.defer()
    prompt = f"""Make a good prompt for DALL-E2 (A Stable diffusion model) to make a picture of this story. Only return the prompt that will be passed in directly: \n\n {story_as_text}"""
    progress_message = await ctx.send(".")

    output_waiting_task = append_dots_to_message_task(progress_message, "Finding a good prompt")

    prompt = await asyncify(ask_gpt)(
        prompt_to_gpt=prompt,
        debug=False,
        u4=False,
    )
    output_waiting_task.cancel()

    ic(prompt)
    content = f"Asking improv gods to visualize - *{prompt}* "

    output_waiting_task = append_dots_to_message_task(progress_message, content)

    response = None
    try:
        response = await asyncify(openai.Image.create)(
            prompt=prompt,
            n=count,
        )
        ic(response)
        image_urls = [response["data"][i]["url"] for i in range(count)]
        ic(image_urls)

        images = []

        for url in image_urls:
            image_data = await download_image(url)
            image_file = discord.File(BytesIO(image_data), filename="image.png")
            images.append(image_file)

        await ctx.followup.send(files=images)
        await progress_message.edit(content=f"**{prompt}** ")
    except InvalidRequestError as e:
        await ctx.followup.send(f"Error: {e}")
    finally:
        output_waiting_task.cancel()


class MentionListener(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return

        # Check if the bot is mentioned
        is_dm = isinstance(message.channel, discord.channel.DMChannel)
        is_mention = self.bot.user in message.mentions
        ic(is_mention, is_dm)
        if is_mention or is_dm:
            await on_message_internal(message)
            return


async def append_dots_to_message_in_background(message, base_text):
    # Stop after 30 seconds - probably nver gonna come back after that.
    for i in range(30 * 2):
        base_text += "."
        await message.edit(base_text)
        await asyncio.sleep(0.5)

def append_dots_to_message_task(message, base_text):
    return asyncio.create_task(append_dots_to_message_in_background(message, base_text))

async def on_message_internal(message):
    if message.author == bot.user:
        return

    # Possibly this was a mention, if so remove that from the message content.
    message_content = message.content.replace(f"<@{bot.user.id}>", "").strip()

    # If user sends '.', let them get a choice of what to write next
    if message_content.strip() == ".":
        await explore(message)
        return

    # TODO: handle all the other slash commands here

    await extend_story_for_bot(message, message_content)
    return


if __name__ == "__main__":
    app()
