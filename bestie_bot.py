#!python3

import asyncio
import datetime
import json
import os

import discord
import psutil
import typer
from icecream import ic

from rich.console import Console
import openai_wrapper

from openai_wrapper import setup_secret
from langchain_openai.chat_models import ChatOpenAI
from langchain import prompts
from typing import TypeVar, Generic
from pydantic import BaseModel

setup_secret()

console = Console()

model = openai_wrapper.setup_gpt()
app = typer.Typer()

u4 = True


T = TypeVar("T")


class BotState(Generic[T]):
    context_to_state = dict()
    defaultState: T

    def __init__(self, defaultState):
        self.defaultState = defaultState

    def __ket_for_ctx(self, ctx):
        is_dm_type = isinstance(ctx, discord.channel.DMChannel)
        is_dm_channel = is_dm_type or ctx.guild is None
        if is_dm_channel:
            return f"DM-{ctx.author.name}-{ctx.author.id}"
        else:
            return f"{ctx.guild.name}-{ctx.channel.name}"

    def get(self, ctx) -> T:
        key = self.__ket_for_ctx(ctx)
        if key not in self.context_to_state:
            self.reset(ctx)

        # return a copy of the story
        return self.context_to_state[key][:]

    def set(self, ctx, state: T):
        key = self.__ket_for_ctx(ctx)
        self.context_to_state[key] = state

    def reset(self, ctx):
        self.set(ctx, self.defaultState)


class BestieState(BaseModel):
    model: int


ic(discord)
bot = discord.Bot()
botState = BotState[BestieState](None)

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
 /help - show this help
 /debug - Dump state
 /set-model - set model to one (tbd - use the UX for this)
 /reset - restart the conversation
When you DM the bot directly, or include a @{bot.user.display_name} in a channel

 - More coming ...
    ```"""


# Due to permissions, we should only get this for a direct message
@bot.event
async def on_message(message):
    ic("bot.on_message", message)
    ic(message.content)


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

    active_story = get_bot_state(ctx)

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
    set_bot_state(ctx, active_story)

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


@bot.command(description="Reset THe bot State")
async def reset(
    ctx,
):
    botState.reset(ctx)
    await smart_send(ctx, "The bot is now reset")


@bot.command(description="Show the story so far, or extend it")
async def story(
    ctx,
    extend: discord.Option(
        str, name="continue_with", description="continue story with", required="False"
    ),
):
    await extend_story_for_bot(ctx, extend)


@bot.command(description="Show help")
async def help(ctx):
    response = f"{bot_help_text}"
    await ctx.respond(response)


@bot.command(description="See local state")
async def debug(ctx):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    debug_out = f"""```ansi
Process:
    Up time: {datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())}
    VM: {memory_info.vms / 1024 / 1024} MB
    Residitent: {memory_info.rss / 1024 / 1024} MB
    TBD do state generically
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
            token = SECRETS["discord-bestie-bot"]

    # throw if token not found
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
    bot.run(token)


async def edit_message_to_append_dots_every_second(message, base_text):
    # Stop after 30 seconds - probably nver gonna come back after that.
    for _ in range(30 * 2):
        base_text += "."
        await message.edit(base_text)
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    app()
