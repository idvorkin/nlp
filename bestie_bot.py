#!python3

import datetime
import os

import discord
import psutil
import typer
from icecream import ic

from rich.console import Console

from discord_helper import BotState, draw_progress_bar, send, get_bot_token
from openai_wrapper import setup_secret
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import bestie

setup_secret()

console = Console()

app = typer.Typer()


class BestieState:
    def __init__(self):
        self.model_name = "2021+3d"
        self.memory = bestie.createBestieMessageHistory()


ic(discord)
bot = discord.Bot()
g_botStateStore = BotState[BestieState](BestieState)
bot_help_text = "Replaced on_ready"


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

    state = g_botStateStore.get(ctx)
    model = ChatOpenAI(model=bestie.models[state.model_name])
    state.memory.add_user_message(message=message_content)
    prompt = ChatPromptTemplate.from_messages(state.memory.messages)
    chain = prompt | model

    progress_bar_task = await draw_progress_bar(ctx)
    result = await chain.ainvoke({})
    progress_bar_task.cancel()
    ai_output = str(result.content)
    ic(ai_output)
    state.memory.add_ai_message(ai_output)
    await send(ctx, f"{ai_output}")


@bot.command(description="Reset the bot State")
async def reset(
    ctx,
):
    g_botStateStore.reset(ctx)
    await ctx.send("The bot is now reset")


@bot.command(description="Show help")
async def help(ctx):
    response = f"{bot_help_text}"
    await ctx.send(response)


@bot.command(description="Set the model")
async def model(ctx, model):
    if model not in bestie.models.keys():
        error = f"model not valid, needs to be one of : {bestie.models.keys()}"
        await send(ctx, error)
        return

    g_botStateStore.reset(ctx)
    state = g_botStateStore.get(ctx)
    state.model_name = model
    await ctx.send(f"model set to {model}")


@bot.command(description="See local state")
async def debug(ctx):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    state = g_botStateStore.get(ctx)
    debug_out = f"""```ansi
Process:
    Up time: {datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())}
    VM: {memory_info.vms / 1024 / 1024} MB
    Residitent: {memory_info.rss / 1024 / 1024} MB
    States: {g_botStateStore.context_to_state.keys()}
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
    bot.run(get_bot_token("discord-bestie-bot"))


if __name__ == "__main__":
    app()
