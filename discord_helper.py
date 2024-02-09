from typing import TypeVar, Generic, Callable
import os
from icecream import ic
import asyncio
from pathlib import Path
import json

T = TypeVar("T")


class BotState(Generic[T]):
    context_to_state = dict()
    defaultStateFactory: Callable[[], T]

    def __init__(self, defaultStateFactory: Callable[[], T]):
        self.defaultStateFactory = defaultStateFactory

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
        ic("setting state", key)
        self.context_to_state[key] = state

    def reset(self, ctx):
        self.set(ctx, self.defaultStateFactory())
        ic("bot reset")


def ctx_to_send_function(ctx):
    is_channel = hasattr(ctx, "channel")
    return ctx.channel.send if is_channel else ctx.send


async def send(ctx, message):
    return await ctx_to_send_function(ctx)(message)


async def draw_progress_bar(ctx):
    progress_message = await send(ctx, ".")
    return asyncio.create_task(
        edit_message_to_append_dots_every_second(progress_message, ".")
    )


def get_bot_token(secret_key):
    # read token from environment variable, or from the secret box, if in neither throw
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        secret_file = Path.home() / "gits/igor2/secretBox.json"
        SECRETS = json.loads(secret_file.read_text())
        token = SECRETS[secret_key]
    return token


async def edit_message_to_append_dots_every_second(message, base_text):
    # Stop after 30 seconds - probably nver gonna come back after that.
    for _ in range(30 * 2):
        base_text += "."
        await message.edit(base_text)
        await asyncio.sleep(0.5)
