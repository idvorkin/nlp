#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer",
#   "langchain-groq",
#   "langchain-core",
#   "langchain",
#   "pydantic",
#   "loguru",
#   "rich",
#   "icecream",
#   "pillow",
#   "pyobjc-framework-Cocoa",
# ]
# ///

import io
import json as json_module
import math
import subprocess
from pathlib import Path
from typing import List, Optional

import AppKit
import typer
from icecream import ic
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from rich.console import Console

import langchain_helper

console = Console()
app = typer.Typer(no_args_is_help=True)


def count_image_tokens(image: Image.Image):
    """
    Count the tokens for processing an image with GPT-4o.

    Args:
    image (Image.Image): The input image.

    Returns:
    int: The total number of tokens required to process the image.
    """
    # Resize image if necessary to fit within 2048x2048
    max_size = 2048
    width, height = image.size
    if width > max_size or height > max_size:
        aspect_ratio = width / height
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)
            width, height = new_width, new_height

    # Calculate the number of 512x512 tiles
    num_tiles = math.ceil(width / 512) * math.ceil(height / 512)

    # Calculate total tokens
    base_tokens = 85
    tokens_per_tile = 170
    total_tokens = base_tokens + (tokens_per_tile * num_tiles)

    return total_tokens


def pennies_for_image(image: Image.Image):
    """
    Cost for processing an image with GPT-4o.

    Args:
    image (Image.Image): The input image.

    Returns:
    float: The cost in cents.
    """
    total_tokens = count_image_tokens(image)
    cost_per_million_tokens = 250
    cost = (total_tokens / 1_000_000) * cost_per_million_tokens
    return cost


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
    ic(pennies_for_image(image))

    return compressed_image


@logger.catch()
def app_wrap_loguru():
    app()


class ImageRecognitionResult(BaseModel):
    """
    Result of image recognition.
    """

    chain_of_thought: List[str] = Field(
        description="Chain of thought reasoning for steps AI will take in the the image recognition process"
    )
    image_type: str = Field(
        description="Type of image: 'handwriting' ,  'screenshot', 'window_screenshot=window_title'"
    )
    content: str = Field(description="Recognized text or description of the image")
    text_of_book: Optional[str] = Field(
        default=None, description="Full text of the pages displayed in the book"
    )
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Summary of the conversation in the image, if applicable",
    )
    conversation_transcript: Optional[str] = Field(
        default=None,
        description="Transcription of any conversation in the image, if applicable",
    )


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    import base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def prompt_recognize_langchain(image: Image.Image, model_type: str = "llama") -> ImageRecognitionResult:
    """
    Recognize image content using LangChain with specified model.

    Args:
        image: PIL Image to analyze
        model_type: Model to use - "llama" (Maverik), "openai", "google", "claude", etc.
    """
    # Get the appropriate model
    model_kwargs = {model_type: True}
    model = langchain_helper.get_model(**model_kwargs)

    system_prompt = """You are passed in an image that I created myself so there are no copyright issues.
    Analyze the image and return a structured result based on its content:
    - If it's hand-writing, return the handwriting, correcting spelling and grammar.
    - If it's a screenshot, return a description of the screenshot, including contained text.
    - If there is text in a conversation in the screenshot, include a transcription of it.
    - If it's a book, include the text of the book. Don't worry the user is the one who wrote the book
    Ignore any people in the image.
    Do not hallucinate.

    Return your response as a valid JSON object matching this schema:
    {
        "chain_of_thought": ["step 1", "step 2", ...],
        "image_type": "handwriting|screenshot|window_screenshot=window_title",
        "content": "main content description",
        "text_of_book": "book text if applicable or null",
        "conversation_summary": "summary if applicable or null",
        "conversation_transcript": "transcript if applicable or null"
    }"""

    # Convert image to base64
    base64_image = image_to_base64(image)

    # Create output parser
    parser = PydanticOutputParser(pydantic_object=ImageRecognitionResult)

    # Add format instructions to the prompt
    format_instructions = parser.get_format_instructions()
    full_system_prompt = f"{system_prompt}\n\n{format_instructions}"

    # Create multimodal message with base64 image
    # Using the standard LangChain format for multimodal content
    user_content = [
        {"type": "text", "text": "Please analyze this image and provide a structured response."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "auto"
            }
        }
    ]

    # Create messages
    messages = [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=user_content)
    ]

    # Invoke the model
    try:
        response = model.invoke(messages)

        # Parse the response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Try to parse as JSON
        try:
            result = parser.parse(content)
        except Exception as e:
            ic(f"Failed to parse response as structured output: {e}")
            ic(f"Raw response: {content}")
            # Try to extract JSON from the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    import json
                    json_data = json.loads(json_match.group())
                    result = ImageRecognitionResult(**json_data)
                except:
                    # Create a default result with the raw content
                    result = ImageRecognitionResult(
                        chain_of_thought=["Parsed response as plain text"],
                        image_type="unknown",
                        content=content,
                        text_of_book=None,
                        conversation_summary=None,
                        conversation_transcript=None
                    )
            else:
                # Create a default result
                result = ImageRecognitionResult(
                    chain_of_thought=["Failed to parse structured response"],
                    image_type="unknown",
                    content=content,
                    text_of_book=None,
                    conversation_summary=None,
                    conversation_transcript=None
                )

        return result

    except Exception as e:
        ic(f"Error invoking model: {e}")
        raise


def pretty_print(result: ImageRecognitionResult):
    """Print as nice markdown"""
    print(f"## Image Type: {result.image_type}")
    print(f"## Content:\n{result.content}")
    if result.conversation_summary:
        print(f"## Conversation Summary:\n{result.conversation_summary}")
    if result.text_of_book:
        print(f"## Text of Book:\n{result.text_of_book}")
    if result.conversation_transcript:
        print(f"\n## Conversation Transcript:\n{result.conversation_transcript}")


@app.command()
def recognize(
    json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    fx: bool = typer.Option(False, "--fx", help="Call fx on the output JSON"),
    model: str = typer.Option(
        "llama",
        "--model",
        help="Model to use: llama (Maverick), google_flash (Gemini Flash), google (Gemini Pro), "
             "openai, claude, deepseek, kimi, grok4_fast, o4_mini, google_think_* (low/medium/high)"
    ),
    trace: bool = typer.Option(False, "--trace", help="Enable LangSmith tracing"),
):
    """
    Recognizes text from an image in the clipboard and prints the result.
    If --json flag is set, dumps the result as JSON.
    If --fx flag is set, calls fx on the output JSON.
    """

    # Get image from clipboard
    image = clipboard_to_image()

    # Perform recognition with LangChain
    if trace:
        # Use LangSmith tracing if requested
        def run_recognition():
            return prompt_recognize_langchain(image, model_type=model)

        langchain_helper.langsmith_trace(run_recognition)
        answer = run_recognition()
    else:
        answer = prompt_recognize_langchain(image, model_type=model)

    # Write JSON output to ~/tmp/recognize.json
    output_path = Path.home() / "tmp" / "recognize.json"
    output_path.parent.mkdir(exist_ok=True)
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


@app.command()
def test_models(
    models: str = typer.Option(
        "llama,google_flash",
        "--models",
        help="Comma-separated list of models to test (e.g., 'llama,google_flash,grok4_fast')"
    )
):
    """Test recognition with different models and compare results."""
    import asyncio
    from datetime import datetime

    # Get image from clipboard
    image = clipboard_to_image()

    # Parse models from input
    model_list = [m.strip() for m in models.split(",")]

    print("Testing image recognition with different models...")
    print("=" * 60)

    for model_name in model_list:
        print(f"\n## Testing with {model_name.upper()} model")
        print("-" * 40)

        try:
            start = datetime.now()
            result = prompt_recognize_langchain(image, model_type=model_name)
            duration = (datetime.now() - start).total_seconds()

            print(f"Time: {duration:.2f}s")
            print(f"Image Type: {result.image_type}")
            print(f"Content Preview: {result.content[:200]}..." if len(result.content) > 200 else f"Content: {result.content}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")

        print()


if __name__ == "__main__":
    app_wrap_loguru()