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
from langchain.prompts import ChatPromptTemplate
from typing import TypeVar, Generic
import bestie

setup_secret()

console = Console()

model = openai_wrapper.setup_gpt()
app = typer.Typer()

u4 = True


T = TypeVar("T")


class BotState(Generic[T]):
    context_to_state = dict()
    defaultState: T

    def __init__(self, defaultStateFactory):
        self.defaultState = defaultStateFactory()

    def __ket_for_ctx(self, ctx):
        ic(type(ctx))
        is_channel = ctx.guild is not None
        if is_channel:
            return f"{ctx.guild.name}-{ctx.channel.name}"
        else:
            return f"DM-{ctx.author.name}-{ctx.author.id}"

    def get(self, ctx) -> T:
        key = self.__ket_for_ctx(ctx)
        if key not in self.context_to_state:
            self.reset(ctx)

        # return a copy of the story
        return self.context_to_state[key]

    def set(self, ctx, state: T):
        key = self.__ket_for_ctx(ctx)
        self.context_to_state[key] = state

    def reset(self, ctx):
        self.set(ctx, self.defaultState)


class BestieState:
    model_name = "2021+3d"
    memory = bestie.createBestieMessageHistory()


ic(discord)
bot = discord.Bot()
botState = BotState[BestieState](BestieState)
bot_help_text = "Replaced on_ready"


def ctx_to_send_function(ctx):
    is_message = not hasattr(ctx, "defer")
    return ctx.channel.send if is_message else ctx.send


async def send(ctx, message):
    return await ctx_to_send_function(ctx)(message)


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    global bot_help_text
    bot_help_text = f"""```

Commands:
 /help - show this help
 /debug - Dump state
 /model - set model to one (tbd - use the UX for this)
 /reset - restart the conversation
When you DM the bot directly, or include a @{bot.user.display_name} in a channel

 - More coming ...
    ```"""


# Due to permissions, we should only get this for a direct message
@bot.event
async def on_message(ctx):
    # if message is from me, skip it
    if ctx.author.bot:
        # ic ("Ignoring message from bot", message)
        return

    ic("bot.on_message", ctx)
    if len(ctx.content) == 0:
        return

    message_content = ctx.content.replace(f"<@{bot.user.id}>", "").strip()

    state = botState.get(ctx)
    model = ChatOpenAI(model=bestie.models[state.model_name])
    state.memory.add_user_message(message=message_content)
    prompt = ChatPromptTemplate.from_messages(state.memory.messages)
    chain = prompt | model
    progress_message = await ctx_to_send_function(ctx)(".")
    output_waiting_task = asyncio.create_task(
        edit_message_to_append_dots_every_second(progress_message, ".")
    )
    result = await chain.ainvoke({})
    output_waiting_task.cancel()
    ai_output = str(result.content)
    ic(ai_output)
    state.memory.add_ai_message(ai_output)
    await send(ctx, f"{ai_output}")


@bot.command(description="Reset THe bot State")
async def reset(
    ctx,
):
    botState.reset(ctx)
    await send(ctx, "The bot is now reset")


@bot.command(description="Show help")
async def help(ctx):
    response = f"{bot_help_text}"
    await send(ctx, response)


@bot.command(description="Set the model")
async def model(ctx, model):
    if model not in bestie.models.keys():
        error = f"model not valid, needs to be one of : {bestie.models.keys()}"
        await send(ctx, error)
        return
    state = botState.get(ctx)
    state.model_name = model
    await send(ctx, f"model set to {model}")


@bot.command(description="See local state")
async def debug(ctx):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    state = botState.get(ctx)
    debug_out = f"""```ansi
Process:
    Up time: {datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())}
    VM: {memory_info.vms / 1024 / 1024} MB
    Residitent: {memory_info.rss / 1024 / 1024} MB
    States: {botState.context_to_state.keys()}
    Model = {state.model_name}
    Current Chat History:
    ```
    """
    # the first is the system message, skip that
    for m in state.memory.messages[1:]:
        debug_out += f"{m}\n"

    # max message is 2000
    debug_out = debug_out[:1900]

    await send(ctx, debug_out)


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
