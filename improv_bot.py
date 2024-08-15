#!python3

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
from discord_helper import draw_progress_bar, get_bot_token, send, BotState
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core import messages

setup_secret()

console = Console()

app = typer.Typer(no_args_is_help=True)

openai_wrapper.setup_secret()
model = ChatOpenAI(temperature=1.0, model=openai_wrapper.gpt4.name)

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


def get_default_story_start():
    return default_story_start


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
 /three-things - play three things
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


class ImprovBotState(BaseModel):
    story: ImprovStory


def new_bot_state():
    return ImprovBotState(story=default_story_start)


g_bot_state = BotState[ImprovBotState](new_bot_state)


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
        g_bot_state.set(self.ctx, self.story)
        await interaction.response.edit_message(content=colored, view=None)


@bot.command(description="Explore alternatives with the bot")
async def explore(ctx):
    active_story = g_bot_state.get(ctx).story
    await ctx.defer()

    colored = color_story_for_discord(active_story)
    # Can pass in a message or a context, silly pycord, luckily can cheat in pycord
    view = View()

    prompt = prompt_gpt_to_return_json_with_story_and_an_additional_fragment_as_json()

    progres_task = await draw_progress_bar(ctx, colored)

    # todo extend this to be n stories
    # n = 4
    list_of_json_version_of_a_story = await asyncify(llm_extend_story)(
        prompt_to_gpt=prompt
    )
    progres_task.cancel()

    # make stories from json
    list_of_stories = [
        json.loads(json_version_of_a_story, object_hook=lambda d: Fragment(**d))
        for json_version_of_a_story in list_of_json_version_of_a_story
    ]

    # write a button for each fragment.
    for story in list_of_stories:
        # add a button for the last fragment of each
        view.add_item(StoryButton(label=story[-1].text[:70], ctx=ctx, story=story))


@bot.command(description="Start a new story with the bot")
async def once_upon_a_time(ctx, start=""):
    g_bot_state.reset(ctx)
    active_story = g_bot_state.get(ctx).story
    story_text = " ".join([f.text for f in active_story])
    if start:
        active_story += [Fragment(player="user", text=start)]
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
    ) | model.bind(
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
    active_story = g_bot_state.get(ctx).story

    if not extend:
        # If called with an empty message lets send help as well
        colored = color_story_for_discord(active_story)
        ic(colored)
        await send(ctx, f"{bot_help_text}\n**The story so far:** {colored}")
        return

    await ctx.defer()

    user_said = Fragment(player=ctx.author.name, text=extend)
    active_story += [user_said]
    ic(active_story)
    colored = color_story_for_discord(active_story)
    result = await llm_extend_story(active_story)
    ic(result)
    active_story = result.Story  # todo clean types up
    g_bot_state.get(ctx).story = active_story

    # convert story to text
    print_story(active_story, show_story=True)
    story_text = " ".join([f.text for f in active_story])
    ic(story_text)
    colored = color_story_for_discord(active_story)
    await send(ctx, colored)
    ic(colored)


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


def prompt_three_things(category=""):
    instructions = """You are an improv coach. Players want to play 3 things. Start by making up a 3 things prompt, and giving an answer, then give the users a 3 things prompt to do them selves. If they give you a category, use it for your example, and for the 3 things you'll ask the user

Other instructions:

- Make the prompts and categories very creative, and funny
- **do not use** dragon or asteroid or wizards pocket unless the user asks.
- Make your examples and prompts really funny, even when given a serious category
- If they give you a catgeory

Here's an example response if user mentions asteroids:
--

**3 things you might need when cleaning an asteroid:**

1. A space suit designed for extreme conditions, ensuring safety from the asteroid's potentially harsh environment.
2. Specialized tools for breaking up, collecting, or analyzing asteroid material, such as a space-grade pickaxe or a laser cutter.
3. A tethering system to keep you anchored to the asteroid, preventing you from floating away into space due to the very low gravity.

Your turn to play...

**3 things a turtle might say if it found itself on an asteroid**

"""

    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=category),
        ]
    )


@bot.command(description="Play 3 things")
async def three_things(ctx, category=""):
    await ctx.defer()
    creative_model = ChatOpenAI(temperature=1.5, model=openai_wrapper.gpt4.name)
    result = await (
        prompt_three_things(category) | creative_model | StrOutputParser()
    ).ainvoke({})
    await ctx.respond(result)


@bot.command(description="Show help")
async def help(ctx):
    active_story = g_bot_state.get(ctx).story
    colored = color_story_for_discord(active_story)
    response = f"{bot_help_text}\n**The story so far:** {colored}"
    await send(ctx, response)


@bot.command(description="See local state")
async def debug(ctx):
    active_story = g_bot_state.get(ctx).story
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
    bot.run(get_bot_token("DISCORD_IMPROV_BOT"))


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@bot.command(description="Visualize the story so far")
async def visualize(ctx, count: int = 2):
    count = min(count, 8)
    active_story = g_bot_state.get(ctx).story
    story_as_text = " ".join([f.text for f in active_story])
    await ctx.defer()
    prompt = f"""Make a good prompt for DALL-E2 (A Stable diffusion model) to make a picture of this story. Only return the prompt that will be passed in directly: \n\n {story_as_text}"""
    output_waiting_task = await draw_progress_bar(ctx, "figuring out prompt")

    chain_make_dalle_prompt = prompts.ChatPromptTemplate.from_messages(
        ("user", prompt)
    ) | ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name)

    r = await chain_make_dalle_prompt.ainvoke({})
    prompt = r.content

    output_waiting_task.cancel()

    ic(prompt)
    content = f"Asking improv gods to visualize - *{prompt}* "

    output_waiting_task = await draw_progress_bar(ctx, content)

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


def prompt_to_flat(m):
    # if m is a of type Systemmesssage return m
    message = {"role": "user", "content": m.content}
    if isinstance(m, messages.SystemMessage):
        message["role"] = "system"
    return message


# @app.command() -- uncomment to be able to export to the prompts
def pf_3t():
    prompt = prompt_three_things()
    import json

    messages = [prompt_to_flat(m) for m in prompt.messages]
    print(json.dumps(messages))


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
