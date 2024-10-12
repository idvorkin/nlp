#!python3

import replicate
import requests
from loguru import logger
from rich.console import Console
import typer
from icecream import ic
from pathlib import Path
import openai_wrapper
import ell
import os

console = Console()
app = typer.Typer(no_args_is_help=True)

ELL_LOGDIR = os.path.expanduser("~/tmp/ell_logdir")
ell.init(store=ELL_LOGDIR, autocommit=True)


@ell.simple(model=openai_wrapper.gpt4.name)
def prompt_illustrate_igor(prompt: str):
    """
    You are an AI that makes great prompts for an image generation model. The user passes in a simple prompt and you make it richer, when they refer to igor, you should say it's idvorkin, who has the charecteristic below.  Only return the prompt as that is what is passed to the image generation model.

      "Charming and whimsical portrait of Idvorkin, a bald, 40 year old 5 foot 8, 175 lbs athletcic build. Captured with a vintage 35mm camera (f/1.8), in vivid 12K resolution. He has a playful look and a twinkle in his eye."
    """
    return prompt


@ell.simple(model=openai_wrapper.gpt4.name)
def prompt_illustrate(prompt: str):
    """
    You are an AI that makes great prompts for an image generation model. The user passes in a simple prompt and you make it richer,
    """
    return prompt


IDVORKIN_LORA_MODEL = "idvorkin/idvorkin-flux-lora-1:4bd4ea7bf781298ebd315871972b9becc6c9e94d3d361bfb2425098e40e88192"


@app.command()
def gen_flux(prompt: str):
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
    print(output)


@app.command()
def gen_igor(prompt: str):
    ic("making prompt for:", prompt)
    lora_prompt = prompt_illustrate_igor(prompt)
    ic(lora_prompt)

    output = replicate.run(
        IDVORKIN_LORA_MODEL,
        input={
            "model": "dev",
            "prompt": lora_prompt,
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
    for image in output:
        print(image)

    ret = make_grid_of_images(output)
    ic(ret)


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
    trainings = replicate.trainings.list()
    for train in trainings:
        ic(train)

    # first model
    first_model = replicate.trainings.get("0wenrm2tcdrm40chj5ytchdebm")
    Path("model_dump.json").write_text(first_model.json(indent=2))


@app.command()
def dump():
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


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
