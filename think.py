#!python3


import sys
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
from langchain_core.pydantic_v1 import BaseModel
from icecream import ic


class GroupOfPoints(BaseModel):
    GroupDescription: str
    Points: List[str]


class AnalyzeArtifact(BaseModel):
    """Each section contains a group of points, there should always be multiple groups when in a list"""

    Summary: List[GroupOfPoints]
    QuestionsToReflectOn: List[GroupOfPoints]
    RelatedTopics: List[GroupOfPoints]


def prompt_think_about_document(diff_output):
    instructions = """
You are a brilliant expert at critical thinking. You help people digest artificats and enhance their thinking.
For each section, return 5 groups of points, with each group containing 5-10 points
You will be passed in a text artifcat
"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=diff_output),
        ]
    )


async def a_think(json: bool, fx: bool):
    llms = [
        langchain_helper.get_model(openai=True),
        # langchain_helper.get_model(claude=True),
        # langchain_helper.get_model(google=True),
    ]

    user_text = "".join(sys.stdin.readlines())

    def do_llm_think(llm) -> List[[AnalyzeArtifact, BaseChatModel]]:  # type: ignore
        return prompt_think_about_document(user_text) | llm.with_structured_output(
            AnalyzeArtifact
        )

    analyzed_artifacts = await langchain_helper.async_run_on_llms(do_llm_think, llms)

    for analysis, llm, duration in analyzed_artifacts:
        import builtins

        if json:
            builtins.print(analysis.json(indent=2))
        if fx:
            # write to temp, and run fx on it
            import tempfile
            import subprocess

            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(analysis.json(indent=2).encode())
            ic(temp.name)
            cmd = f"fx {temp.name}"
            ic(cmd)
            subprocess.run(cmd, shell=True)
        else:
            builtins.print(analysis.json(indent=2))
            print(
                f"# -- model: {langchain_helper.get_model_name(llm)} | {duration.total_seconds():.2f} seconds --"
            )
            print(analysis)


console = Console()
app = typer.Typer()


@app.command()
def think(
    trace: bool = False,
    json: bool = False,
    fx: bool = False,
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_think(json, fx))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
