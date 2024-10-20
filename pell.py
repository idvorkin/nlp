#!python3
# pylint: disable=missing-function-docstring


import typer
from icecream import ic
from rich.console import Console
from loguru import logger
import ell
import os
import openai
import openai_wrapper
from PIL import Image, ImageDraw

console = Console()
app = typer.Typer(no_args_is_help=True)
openai_client = openai.Client()


# Define ELL_LOGDIR as a constant
ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")

ell.init(store=ELL_LOGDIR, autocommit=True)
ell.models.groq.register()


@app.command()
def list_models():
    console.print("Models available:")
    # console.print(ell.models.groq.


# A fine tune I created
igor_model = "ft:gpt-4o-mini-2024-07-18:idvorkinteam:i-to-a-3d-gt-2021:9qiMMqOz"


@ell.simple(model=igor_model, client=openai_client)
def hello(world: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


# @ell.simple(model="llama-3.2-90b-vision-preview")


@ell.simple(model=openai_wrapper.get_ell_model(llama=True))
def hello_groq(world: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


@ell.complex(model=openai_wrapper.get_ell_model(llama=True))  # type: ignore
def hello_groq_image(image: Image.Image):
    system = """
    You are passed in an image that I created myself so there are no copyright issues, describe what is in it
    """
    return [ell.user(system), ell.user([image])]  # type: ignore


@app.command()
def scratch():
    response = hello("Igor")
    ic(response)


@app.command()
def groq():
    # Call hello_groq function with "Igor" and print the response
    response = hello_groq("Igor")
    ic(response)

    # Create an image with 4 rectangles and pass it to hello_groq_image function
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 99, 99], fill=(255, 0, 0))
    draw.rectangle([100, 0, 199, 99], fill=(0, 255, 0))
    draw.rectangle([0, 100, 99, 199], fill=(0, 0, 255))
    draw.rectangle([100, 100, 199, 199], fill=(255, 255, 0))
    response2 = hello_groq_image(img)
    ic(response2)


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
