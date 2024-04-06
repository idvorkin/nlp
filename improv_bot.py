#!python3

import asyncio
import datetime
import json
import os
import random
from io import BytesIO
from typing import List

import aiohttp
import discord
import openai
import psutil
import typer
from asyncer import asyncify
from discord.ext import commands
from discord.ui import View
from icecream import ic

from pydantic import BaseModel
from rich.console import Console
from rich.text import Text
import openai_wrapper

from openai_wrapper import setup_secret
from langchain_openai.chat_models import ChatOpenAI
from langchain import prompts
import discord_helper

setup_secret()

console = Console()

model = openai_wrapper.setup_gpt()
app = typer.Typer()

u4 = True


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


ImprovStory = List[Fragment]

default_story_start = [
    Fragment.Pos("coach", "Once upon a time", "A normal story start"),
]


class AppendCoachFragmentThenOuputStory(BaseModel):
    Story: ImprovStory


def print_story(story: ImprovStory, show_story: bool):
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


def prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json():
    return f"""

### Instructions

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

* I'll write 1-5 words, and then you do the same, and we'll go back and forth writing the story.
* You will add a reasoning to why you added the next fragment to the story
* You can correct spelling and capilization mistakes

### Techincal Details

* The users input will look like a python represenation
* You should provide output by taking the input, adding your own fragment as the coach, then calling the function

### Examples

In these examples, you'd have called the function with the example output

Example 1 Input:

{example_1_in}

Example 1 What you'd call the Extend function with

{example_1_out}
--

Example 2 Input:

{example_2_in}

Example 2 What you'd call the Extend function with

{example_2_out}

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


def get_story_for_channel(ctx) -> ImprovStory:
    key = key_for_ctx(ctx)
    if key not in context_to_story:
        reset_story_for_channel(ctx)

    # return a copy of the story
    return context_to_story[key][:]


def set_story_for_channel(ctx, story):
    key = key_for_ctx(ctx)
    context_to_story[key] = story


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
    # This is part of the "meat" of the game logic.
    async def callback(self, interaction: discord.Interaction):
        colored = color_story_for_discord(self.story)
        set_story_for_channel(self.ctx, self.story)
        await interaction.response.edit_message(content=colored, view=None)


@bot.command(description="Explore alternatives with the bot")
async def explore(ctx):
    active_story = get_story_for_channel(ctx)
    is_message = not hasattr(ctx, "defer")

    if not is_message:
        await ctx.defer()

    colored = color_story_for_discord(active_story)
    # Can pass in a message or a context, silly pycord, luckily can cheat in pycord
    progress_message = await smart_send(ctx, ".")
    view = View()

    prompt = prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json()

    output_waiting_task = asyncio.create_task(
        edit_message_to_append_dots_every_second(progress_message, colored)
    )

    # todo extend this to be n stories
    # n = 4
    list_of_json_version_of_a_story = await asyncify(llm_extend_story)(
        prompt_to_gpt=prompt
    )
    output_waiting_task.cancel()

    # make stories from json
    list_of_stories = [
        json.loads(json_version_of_a_story, object_hook=lambda d: Fragment(**d))
        for json_version_of_a_story in list_of_json_version_of_a_story
    ]

    # write a button for each fragment.
    for story in list_of_stories:
        # add a button for the last fragment of each
        view.add_item(StoryButton(label=story[-1].text[:70], ctx=ctx, story=story))
    await progress_message.edit(content=colored, view=view)
    if not is_message:
        # acknolwedge without sending
        await ctx.send(content="")


@bot.command(description="Start a new story with the bot")
async def once_upon_a_time(ctx):
    reset_story_for_channel(ctx)
    active_story = get_story_for_channel(ctx)
    story_text = " ".join([f.text for f in active_story])
    ic(story_text)
    colored = color_story_for_discord(active_story)
    response = f"{bot_help_text}\n**The story so far:** {colored}"
    await ctx.respond(response)


async def llm_extend_story(active_story):
    extendStory = openai_wrapper.openai_func(AppendCoachFragmentThenOuputStory)
    system_prompt = (
        prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json()
    )
    ic(system_prompt)

    chain = prompts.ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", str(active_story))]
    ) | ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name).bind(
        tools=[extendStory], tool_choice=openai_wrapper.tool_choice(extendStory)
    )
    r = await chain.ainvoke({})
    extended_story_json_str = r.additional_kwargs["tool_calls"][0]["function"][
        "arguments"
    ]
    return AppendCoachFragmentThenOuputStory.model_validate(
        json.loads(extended_story_json_str)
    )


async def extend_story_for_bot(ctx, extend: str = ""):
    # if story is empty, then start with the default story
    ic(extend)
    is_message = not hasattr(ctx, "defer")

    active_story = get_story_for_channel(ctx)

    if not extend:
        # If called with an empty message lets send help as well
        colored = color_story_for_discord(active_story)
        ic(colored)
        await smart_send(ctx, f"{bot_help_text}\n**The story so far:** {colored}")
        return

    if not is_message:
        await ctx.defer()

    user_said = Fragment(player=ctx.author.name, text=extend)
    active_story += [user_said]
    ic(active_story)
    ic("calling gpt")
    colored = color_story_for_discord(active_story)
    # print progress in the background while running
    progress_message = (
        await ctx.channel.send(".") if is_message else await ctx.send(".")
    )
    output_waiting_task = asyncio.create_task(
        edit_message_to_append_dots_every_second(progress_message, f"{colored}")
    )

    result = await llm_extend_story(active_story)
    output_waiting_task.cancel()
    ic(result)
    active_story = result.Story  # todo clean types up
    set_story_for_channel(ctx, active_story)

    # convert story to text
    print_story(active_story, show_story=True)
    story_text = " ".join([f.text for f in active_story])
    ic(story_text)
    colored = color_story_for_discord(active_story)
    ic(colored)

    await progress_message.edit(content=colored)
    if not is_message:
        # acknolwedge without sending
        await ctx.send(content="")


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
Active Story:
    {[repr(f) for f in active_story] }
Other Stories
{context_to_story.keys()}
    ```
    """
    await ctx.respond(debug_out)


@app.command()
def run_bot():
    bot.run(discord_helper.get_bot_token("DISCORD_IMPROV_BOT"))


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
    output_waiting_task = asyncio.create_task(
        edit_message_to_append_dots_every_second(
            progress_message, "Figuring out prompt"
        )
    )

    chain_make_dalle_prompt = prompts.ChatPromptTemplate.from_messages(
        ("user", prompt)
    ) | ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name)

    r = await chain_make_dalle_prompt.ainvoke({})
    prompt = r.content

    output_waiting_task.cancel()

    ic(prompt)
    content = f"Asking improv gods to visualize - *{prompt}* "
    output_waiting_task = asyncio.create_task(
        edit_message_to_append_dots_every_second(progress_message, f"{content}")
    )

    response = None
    try:
        response = await asyncify(openai.images.generate)(
            model="dall-e-3",
            prompt=prompt,
            n=1,
        )
        ic(response)
        images = response.data
        ic(images)
        image_urls = [image.url for image in images]

        # [response["data"][i]["url"] for i in range(count)]
        ic(image_urls)

        images = []

        for url in image_urls:
            image_data = await download_image(url)
            image_file = discord.File(BytesIO(image_data), filename="image.png")
            images.append(image_file)

        await ctx.followup.send(files=images)
        await progress_message.edit(content=f"**{prompt}** ")
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
            await on_mention(message)
            return
        # check if message is a DM

    # TODO: Refactor to be with extend_story_for_bot


async def edit_message_to_append_dots_every_second(message, base_text):
    # Stop after 30 seconds - probably nver gonna come back after that.
    for i in range(30 * 2):
        base_text += "."
        await message.edit(base_text)
        await asyncio.sleep(0.5)


async def on_mention(message):
    if message.author == bot.user:
        return
    message_content = message.content.replace(f"<@{bot.user.id}>", "").strip()

    # If user sends '.', let them get a choice of what to write next
    if message_content.strip() == ".":
        await explore(message)
        return

    # TODO: handle help now

    await extend_story_for_bot(message, message_content)
    return


if __name__ == "__main__":
    app()
