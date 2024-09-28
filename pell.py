#!python3
# pylint: disable=missing-function-docstring


import typer
from icecream import ic
from rich.console import Console
from loguru import logger
import ell
import os

console = Console()
app = typer.Typer(no_args_is_help=True)

# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)
ell.models.groq.register()


@app.command()
def list_models():
    console.print("Models available:")
    # console.print(ell.models.groq.


@ell.simple(model="gpt-4o-mini")
def hello(world: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


# @ell.simple(model="llama-3.2-90b-vision-preview")
@ell.simple(model="llama-3.2-11b-vision-preview")
def hello_groq(world: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


@app.command()
def scratch():
    response = hello_groq("Igor", api_params=dict(n=1))
    ic(response)


@app.command()
def studio():
    import subprocess
    import socket
    import asyncio

    def is_port_in_use(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0

    def open_browser():
        subprocess.run(["open", "http://127.0.0.1:5555"])

    if not is_port_in_use(5555):

        async def run_server_and_open_browser():
            # Start the server asynchronously
            server_process = await asyncio.create_subprocess_exec(
                "ell-studio", "--storage", ELL_LOGDIR
            )

            # Wait for 2 seconds
            await asyncio.sleep(2)

            # Open the browser
            open_browser()

            # Keep the server running
            await server_process.wait()

        # Run the async function
        asyncio.run(run_server_and_open_browser())
    else:
        ic("Studio is already running")
        open_browser()


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
