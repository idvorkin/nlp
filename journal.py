#!python3

import openai

import typer
import ell
import rich
import os

app = typer.Typer(no_args_is_help=True)

# Create custom client with Gemini endpoint
client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

model_to_use = "gemini-1.5-flash"
# Register the model with your custom client
ell.config.register_model(model_to_use, client)


@ell.simple(model=model_to_use, temperature=0.7)
def prompt_hello():
    return "Hello world tell me a joke"


@app.command()
def smoke_test():
    response = prompt_hello()
    rich.print(response)


if __name__ == "__main__":
    app()
