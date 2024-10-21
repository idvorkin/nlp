#!python3

import replicate
import requests
from loguru import logger
from rich.console import Console
import typer
from icecream import ic
from pathlib import Path
import os
import ell
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()


@ell.simple(model=get_ell_model(openai=True))
def prompt_illustrate_igor(prompt: str):
    """
    You are an AI that creates enhanced prompts for an image generation model. The user provides a simple prompt, and you enrich it. When they mention Igor, refer to him as Idvorkin with the following characteristics:

    "Charming and whimsical portrait of Idvorkin, a bald, 40-year-old man standing 5 feet 8 inches tall, weighing 175 lbs with an athletic build. Captured with a vintage 35mm camera (f/1.8), in vivid 12K resolution. He has a playful look and a twinkle in his eye."

    Incorporate these details seamlessly into the enhanced prompt. Return only the enriched prompt, as it will be directly passed to the image generation model.
    """
    return prompt


@ell.simple(model=get_ell_model(openai=True))
def prompt_illustrate(prompt: str):
    """
    You are an AI that makes great prompts for an image generation model. The user passes in a simple prompt and you make it richer,
    """
    return prompt


IDVORKIN_LORA_MODEL = "idvorkin/idvorkin-flux-lora-1:4bd4ea7bf781298ebd315871972b9becc6c9e94d3d361bfb2425098e40e88192"


def show(img):
    if "TMUX" in os.environ:
        os.system(f"timg --grid 4 --title -ps {img}")
    else:
        os.system(f"timg --grid 4 --title -p {img}")


@app.command()
def gen_flux(prompt: str):
    """
    Generate an image using the Flux model based on the given prompt.

    Args:
        prompt (str): The text prompt to generate the image from.
    """
    ic(prompt)
    augmented_prompt = prompt_illustrate(prompt)
    ic(augmented_prompt)

    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={
            "prompt": augmented_prompt,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "safety_tolerance": 2,
            "prompt_upsampling": True,
        },
    )
    ic(output)
    show(output)
    ic(output)


@app.command()
def gen_igor(prompt: str):
    """
    Generate an image of Igor (Idvorkin) using a custom LoRA model based on the given prompt.

    Args:
        prompt (str): The text prompt to generate the image from.
    """
    ic(prompt)
    augmented_prompt = prompt_illustrate_igor(prompt)
    ic(augmented_prompt)

    images = replicate.run(
        IDVORKIN_LORA_MODEL,
        input={
            "model": "dev",
            "prompt": augmented_prompt,
            "lora_scale": 1,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "guidance_scale": 3.5,
            "output_quality": 80,
            "extra_lora_scale": 0.8,
            "num_inference_steps": 28,
            "disable_safety_checker": True,
        },
    )
    image = images[0]

    ic(image)
    show(image)
    ic(image)


def make_grid_of_images(images):
    from PIL import Image
    import requests
    from io import BytesIO
    import tempfile

    # Calculate the grid dimensions
    num_images = len(images)
    grid_size = int(num_images**0.5)
    if grid_size**2 < num_images:
        grid_size += 1

    # Create a new blank image for the grid
    grid_width = grid_size * 300
    grid_height = grid_size * 300
    grid = Image.new("RGB", (grid_width, grid_height))

    # Download and append each image to the grid
    for i, url in enumerate(images):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((300, 300))
        row = i // grid_size
        col = i % grid_size
        grid.paste(img, (col * 300, row * 300))

    # Save the grid to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp_file:
        grid.save(tmp_file.name, format="WEBP")
        return tmp_file.name


@app.command()
def training():
    """
    List and display information about model trainings.
    """
    trainings = replicate.trainings.list()
    for train in trainings:
        ic(train)

    # first model
    first_model = replicate.trainings.get("0wenrm2tcdrm40chj5ytchdebm")
    Path("model_dump.json").write_text(first_model.json(indent=2))


@app.command()
def dump():
    """
    Dump model information and download model files for the custom Idvorkin LoRA model.
    """
    model = replicate.models.get("idvorkin/idvorkin-flux-lora-1")
    # write model to file
    ic(model)
    version = model.versions.list()[0]
    Path("model_dump.json").write_text(version.json(indent=2))

    # Get the URLs of the model files
    files = (
        version.openapi_schema.get("components", {})
        .get("schemas", {})
        .get("Input", {})
        .get("properties", {})
        .get("weights", {})
        .get("default", [])
    )
    ic(files)

    # Download each file
    for file_url in files:
        response = requests.get(file_url)
        filename = file_url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(response.content)


@app.command()
def studio(port: int = Option(None, help="Port to run the ELL Studio on")):
    """
    Launch the ELL Studio interface for interactive model exploration and testing.

    This command opens the ELL Studio, allowing users to interactively work with
    language models, test prompts, and analyze responses in a user-friendly environment.
    """
    run_studio(port=port)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
