#!python3

import openai
import typer
import ell
import rich
import os
from pathlib import Path
from icecream import ic

app = typer.Typer(no_args_is_help=True)

# Create custom client with Gemini endpoint
client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

# model_to_use = "gemini-1.5-flash-002"
model_to_use = "gemini-1.5-pro-002"
# Register the model with your custom client
ell.config.register_model(model_to_use, client)


@ell.simple(model=model_to_use, temperature=0.7)
def prompt_hello():
    return "Hello world tell me a joke"


# Annoying, ell can't take a base64 input of a file, lets use gemini raw for that
gemini_prompt = """You are an expert at transcribing handwritten text. 
Given a pdf of handwritten text, transcribe it accurately while maintaining:

- Original formatting
- Any bullet points or numbering
- When unsure of a word, indicate with [guess], when not legible indicate with [illegible]

- When a new page starts, include a horizontal line, then Page: X of N, then another horizontal line
E.g. 
___
    Page: 1 of 10
___

- Make sure tables are nicely formatted  

E.g. 

| Column 1  |  Column 2 | Column 3  |
| --------- | --------- | --------- |
|A          | B         | C         |
|Hello World| 123       | 456       |

- When parts are illegible, indicate with [illegible].
- When you're less sure about a word, indicate with [guess]
- If date is on first row, include it in the page title
- Merge multiple lines into a single line if they are part of the same paragraph, smart wrapping at 120 char mark
- Fix spelling mistakes
- If there seem to be headings/lists throughout the doc

At the end of the document, include an analysis section it should include: 
- Summary of the document
- Any key insights
- Any tasks that need to be completed
- Any other actions that need to be taken
- For anything that looks like a list spread throughout the document, coalesce into a joint set of items
- E.g. 
    YAB: blah
    other text
    YAB: Bloop
    
    The list becomes
    
    YAB: 
        - blah
        - Bloop
    
    """

# model_to_use = "gemini-1.5-flash-002"
# model_to_use = "gemini-1.5-pro-002"


def gemini_transcribe(pdf_path: str):
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    pdf_data = Part.from_data(
        mime_type="application/pdf", data=Path(pdf_path).read_bytes()
    )

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    vertexai.init(project="tranquil-hawk-325816", location="us-central1")
    model = GenerativeModel(model_name=model_to_use)
    chat = model.start_chat()
    ic("starting", model_to_use)
    response = chat.send_message(
        [gemini_prompt, pdf_data], generation_config=generation_config
    )
    ic(response.usage_metadata)
    return response.text


@app.command()
def transcribe(pdf: str = typer.Argument(..., help="Path to pdf file to transcribe")):
    """Transcribe handwritten text from an image file"""
    # Expand user path if needed
    full_path = os.path.expanduser(pdf)

    if not os.path.exists(full_path):
        rich.print(f"[red]Error: File not found: {full_path}")
        raise typer.Exit(1)

    response = gemini_transcribe(full_path)
    rich.print(response)


if __name__ == "__main__":
    app()
