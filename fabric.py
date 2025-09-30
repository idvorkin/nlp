#!python3


from pathlib import Path
import asyncio
from typing import List

from langchain_core import messages
from langchain_core.language_models import BaseChatModel

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
from icecream import ic
from openai_wrapper import num_tokens_from_string


def prompt_think_about_document(document, prompt):
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=prompt),
            messages.HumanMessage(content=document),
        ]
    )


console = Console()
app = typer.Typer(no_args_is_help=True)


async def a_fabric(
    command: str = "none",
    gist: bool = True,
    path: str = typer.Argument(None),
):
    # check if is installed
    fabric_path = Path("~/rare_gits/fabric/patterns").expanduser()
    if not fabric_path.exists():
        print("Fabric not installed, please install it first")
        print("git clone https://github.com/danielmiessler/fabric ~/rare_gits/fabric")
        return

    pattern = fabric_path / command / "system.md"
    if not pattern.exists():
        print("Not a valid pattern")
        # list all files in fabric_path
        for file in fabric_path.glob("*"):
            print(file.name)
        return

    llms = langchain_helper.get_models(claude=True)

    user_text = langchain_helper.get_text_from_path_or_stdin(path)
    tokens = num_tokens_from_string(user_text)

    if tokens < 8000:
        # only add Llama if the text is small
        llms += [langchain_helper.get_model(llama=True)]

    if tokens < 4000:
        # Add it twice if it fits
        llms += [langchain_helper.get_model(llama=True)]

    thinking_about = f"*Thinking about {path}.*" if path else ""
    ic("thinking", command, tokens)
    header = f"""
*🧠 via [fabric.py](https://github.com/idvorkin/nlp/blob/main/fabric.py) - using [{command}](https://github.com/danielmiessler/fabric/blob/main/patterns/{command}/system.md).*
{thinking_about}
    """

    def do_llm_think(llm) -> List[[str, BaseChatModel]]:  # type: ignore
        from langchain.schema.output_parser import StrOutputParser

        # return prompt_think_about_document(user_text) | llm.with_structured_output( AnalyzeArtifact)
        return (
            prompt_think_about_document(user_text, prompt=pattern.read_text())
            | llm
            | StrOutputParser()
        )

    analyzed_artifacts = await langchain_helper.async_run_on_llms(do_llm_think, llms)

    body = ""
    for analysis, llm, duration in analyzed_artifacts:
        body += f"""
<details>
<summary>

# -- model: {langchain_helper.get_model_name(llm)} | {duration.total_seconds():.2f} seconds --

</summary>

{analysis}

</details>
"""
    output_text = header + "\n" + body
    output_path = Path(
        "~/tmp/fabric.md"
    ).expanduser()  # get smarter about naming these.
    output_path.write_text(output_text)
    ic(output_path)
    if gist:
        # create temp file and write print buffer to it
        langchain_helper.to_gist(output_path)
    else:
        print(output_text)


@app.command()
def fabric(
    command: str = typer.Argument("none"),
    trace: bool = False,
    gist: bool = True,
    path: str = typer.Argument(None),
):
    # check if command file exists, if not print the list
    langchain_helper.langsmith_trace_if_requested(
        trace,
        lambda: asyncio.run(
            a_fabric(
                command,
                gist=gist,
                path=path,
            )
        ),
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
