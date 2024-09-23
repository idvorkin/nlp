#!python3

import io

from PIL import Image
import typer
import ell
import AppKit
import os
from pydantic import BaseModel, Field
from typing import Optional


from loguru import logger
from rich.console import Console
from icecream import ic

# import openai_wrapper
from pathlib import Path
import subprocess

console = Console()
app = typer.Typer(no_args_is_help=True)

ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")
ell.init(store=ELL_LOGDIR, autocommit=True)


def get_image_from_clipboard() -> Image.Image | None:
    pb = AppKit.NSPasteboard.generalPasteboard()  # type: ignore
    data_type = pb.availableTypeFromArray_([AppKit.NSPasteboardTypeTIFF])  # type: ignore
    if data_type:
        data = pb.dataForType_(data_type)  # type: ignore
        return Image.open(io.BytesIO(data))
    return None


def clipboard_to_image(max_width=2000, quality=85):
    image = get_image_from_clipboard()
    if image is None:
        raise ValueError("No image found in clipboard")
    ic(len(image.tobytes()) / 1024 / 1024)

    # Resize the image, into the same same aspect ratio
    original_size_mb = len(image.tobytes()) / 1024 / 1024
    width, height = image.size
    aspect_ratio = width / height
    new_width = min(max_width, width)
    new_height = int(new_width / aspect_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    ic(width, height, new_width, new_height)

    # Convert the image to WebP format with lossy compression
    with io.BytesIO() as output:
        image.save(output, format="WEBP", quality=quality)
        output.seek(0)
        compressed_image = Image.open(output)

    compressed_size_mb = len(compressed_image.tobytes()) / 1024 / 1024
    ic(len(compressed_image.tobytes()) / 1024 / 1024)
    ic(original_size_mb - compressed_size_mb)

    return compressed_image


@logger.catch()
def app_wrap_loguru():
    app()


class ImageRecognitionResult(BaseModel):
    """
    Result of image recognition.
    """

    image_type: str = Field(description="Type of image: 'handwriting' or 'screenshot'")
    content: str = Field(description="Recognized text or description of the image")
    conversation_transcript: Optional[str] = Field(
        default=None,
        description="Transcription of any conversation in the image, if applicable",
    )


@ell.complex(model="gpt-4o-2024-08-06", response_format=ImageRecognitionResult)  # type: ignore
def prompt_recognize(image: Image.Image):
    system = """
    You are passed in an image that I created myself so there are no copyright issues.
    Analyze the image and return a structured result based on its content:
    - If it's hand-writing, return the handwriting, correcting spelling and grammar.
    - If it's a screenshot, return a description of the screenshot, including contained text.
    - If there is text in a conversation in the screenshot, include a transcription of it.
    Ignore any people in the image.
    Do not hallucinate.
    """
    return [ell.system(system), ell.user([image])]  # type: ignore


def pretty_print(result: ImageRecognitionResult):
    """Print as nice markdown"""
    print(f"## Image Type: {result.image_type}")
    print(f"## Content:\n{result.content}")
    if result.conversation_transcript:
        print(f"\n## Conversation Transcript:\n{result.conversation_transcript}")


@app.command()
def recognize(
    json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    fx: bool = typer.Option(False, "--fx", help="Call fx on the output JSON"),
):
    """
    Recognizes text from an image in the clipboard and prints the result.
    If --json flag is set, dumps the result as JSON.
    If --fx flag is set, calls fx on the output JSON.
    """
    import json as json_module

    answer: ImageRecognitionResult = prompt_recognize(clipboard_to_image()).parsed  # type: ignore
    # Write JSON output to ~/tmp/recognize.json
    output_path = Path.home() / "tmp" / "recognize.json"
    with output_path.open("w", encoding="utf-8") as f:
        json_module.dump(answer.model_dump(), f, indent=2)
    ic(f"JSON output written to {output_path}")
    if json or fx:
        json_output = answer.model_dump_json(indent=2)
        if fx:
            subprocess.run(["fx"], input=json_output.encode(), shell=True)
        else:
            print(json_output)
    else:
        pretty_print(answer)


if __name__ == "__main__":
    app_wrap_loguru()
