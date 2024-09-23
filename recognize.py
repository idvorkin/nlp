#!python3

import io

from PIL import Image
import typer
import ell
import AppKit


from loguru import logger
from rich.console import Console
from icecream import ic
# import openai_wrapper

console = Console()
app = typer.Typer(no_args_is_help=True)


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

    ic(len(compressed_image.tobytes()) / 1024 / 1024)
    saved_from_compression_mb = (
        (len(image.tobytes()) - len(compressed_image.tobytes())) / 1024 / 1024
    )
    ic(saved_from_compression_mb)

    return compressed_image


@logger.catch()
def app_wrap_loguru():
    app()


@ell.simple(model="gpt-4o")
def prompt_recognize(image: Image.Image):
    """
    You are passed in an image that I created myself so there are no copyright issues.
    Depend on the image, you need to do different things:
    If it's hand-writing, return the handwriting, correcting spelling and grammar.
    If it's a screenshot, return a description of the screenshot, including contained text
    Ignore any people in the image
    Do not hallucinate
    """
    return [
        ell.user([image])  # type: ignore
    ]


@app.command()
def recognize():
    """
    Recognizes text from an image in the clipboard and prints the result.
    """
    answer = prompt_recognize(clipboard_to_image())
    print(answer)


if __name__ == "__main__":
    app_wrap_loguru()
