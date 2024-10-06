#!python3

import replicate
import requests
from loguru import logger
from rich.console import Console
import typer
from icecream import ic
from pathlib import Path

console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def run():
    output = replicate.run(
        "idvorkin/idvorkin-flux-lora-1:4bd4ea7bf781298ebd315871972b9becc6c9e94d3d361bfb2425098e40e88192",
        input={
            "model": "dev",
            "prompt": "Hyperrealistic photograph of a bald, middle-aged man named Idvorkin, wearing a costume inspired by Starfire from the Teen Titans. He is dressed in a vibrant, metallic purple bodysuit with green accents, including matching gloves and boots. He also wears a belt with a glowing green gem in the center and has his colorful glasses on, adding a unique twist to the look. The background is a futuristic cityscape at night, with tall buildings and neon lights. Idvorkin is striking a powerful pose, as if ready to unleash energy blasts, with a confident and heroic expression. The overall mood is dynamic and otherworldly. Shot with a 50mm lens (f/2.8), rendered in 16K resolution. Realistic skin texture, detailed costume, and a vibrant, sci-fi atmosphere. Ensure the content is safe",
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
    print(output)


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
