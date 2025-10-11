#!python3
# pylint: disable=missing-function-docstring


import typer
from typer import Option
from icecream import ic
from rich.console import Console
from loguru import logger
import ell
import openai
from PIL import Image, ImageDraw
from ell_helper import get_ell_model, init_ell, run_studio
from pydantic import BaseModel

console = Console()
app = typer.Typer(no_args_is_help=True)
openai_client = openai.Client()

# Initialize ELL
init_ell()


@app.command()
def studio(port: int = Option(None, help="Port to run the ELL Studio on")):
    """
    Launch the ELL Studio interface for interactive model exploration and testing.

    This command opens the ELL Studio, allowing users to interactively work with
    language models, test prompts, and analyze responses in a user-friendly environment.
    """
    run_studio(port=port)


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


@ell.simple(model=get_ell_model(llama=True))
def prompt_hello_groq(world: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = world.capitalize()
    return f"Say hello to {name}!"  # User prompt


@ell.complex(model=get_ell_model(llama_vision=True))  # type: ignore
def prompt_recognize_groq_image(image: Image.Image):
    system = """
    You are passed in an image that I created myself so there are no copyright issues, describe what is in it
    """
    return [ell.user(system), ell.user([image])]  # type: ignore


@app.command()
def scratch():
    response = hello("Igor")
    ic(response)


@ell.simple(model=get_ell_model(claude=True), max_tokens=4000)  # type: ignore
def prompt_hello_claude(name: str):
    """You are a unhelpful assistant, make your answers spicy"""  # System prompt
    name = name.capitalize()
    return f"Say hello to {name}!"  # User prompt


@app.command()
def claude(name=typer.Argument("Claude", help="Name to greet")):
    # Call prompt_hello_claude function with the provided name and print the response
    response = prompt_hello_claude(name)
    console.print(response)


@app.command()
def groq():
    # Call hello_groq function with "Igor" and print the response
    response = prompt_hello_groq("Igor")
    ic(response)

    # Create an image with 4 rectangles and pass it to hello_groq_image function
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 99, 99], fill=(255, 0, 0))
    draw.rectangle([100, 0, 199, 99], fill=(0, 255, 0))
    draw.rectangle([0, 100, 99, 199], fill=(0, 0, 255))
    draw.rectangle([100, 100, 199, 199], fill=(255, 255, 0))
    response2 = prompt_recognize_groq_image(img)
    ic(response2)


class JokeWithReasoning(BaseModel):
    ReasonItsFunny: str
    ReasonItsSpicy: str
    Joke: str


@ell.complex(
    model=get_ell_model(openai=True),
    response_format=JokeWithReasoning,
    max_tokens=4000,
)
def prompt_joke_with_reasoning(joke_topic):
    system = """
    Tell a  spicy joke about the topic the user says
    """
    return [ell.user(system), ell.user(joke_topic)]


@app.command()
def groq_models():
    import requests
    import os
    # curl https://api.groq.com/openai/v1/models \ -H "Authorization: Bearer $GROQ_API_KEY"jkjf0w
    response = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
    models = response.json()["data"]
    for model in models:
        ic(model["id"])

@app.command()
def joke(topic: str = typer.Argument("chickens", help="Topic for the joke")):
    response = prompt_joke_with_reasoning(topic)
    joke: JokeWithReasoning = response.content[0].parsed  # type: ignore
    print(joke.Joke)
    ic(joke.ReasonItsFunny)
    ic(joke.ReasonItsSpicy)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
