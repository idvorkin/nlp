#!python3


from pathlib import Path
import sys
import asyncio
from typing import List

from langchain_core import messages
from langchain_core.language_models import BaseChatModel
import requests

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich import print
from rich.console import Console
import langchain_helper
from icecream import ic
from openai_wrapper import num_tokens_from_string
import html2text

# class GroupOfPoints(BaseModel):
# GroupDescription: str
# Points: List[str]


# class AnalyzeArtifact(BaseModel):
# """Each section contains a list of group of points, there should always be 2 or more elements in each list"""

# Summary: List[GroupOfPoints]
# QuestionsToReflectOn: List[GroupOfPoints]
# RelatedTopics: List[GroupOfPoints]


def prompt_think_about_document(document):
    instructions = """
You are a brilliant expert at critical thinking, specialized in digesting and enhancing understanding of various artifacts. The user will rely on you to help them think critically about the thing they are reading.

For this task, you will analyze the provided artifact. Your aim is to structure your analysis into the sections listed below.  Each section should contain between 2 and 5 groups of points. Each group should include 2 to 10 specific points that are critical to understanding the artifact.

Please format your analysis as follows (do not use the word group, but use the actual group or topic), use markdown:

## Summary

### Group:
 - Point 1
 - Point 2
 - ...
### Group:
 - Point 1
 - Point 2
 - ...
   - ...

## Implications and Impact

### Group:
 - Point 1
 - Point 2
 - ...
### Group:
 - Point 1
 - Point 2
 - ...
   - ...


## Critical Assumptions and Risks

[as above]

## Reflection Questions


[as above]

## Contextual Background

[as above]

## Related Topics

[as above]

Ensure that you consider the type of artifact you are analyzing. For instance, if the artifact is a conversation, include points and questions that cover different perspectives and aspects discussed during the conversation.

Ensure that you consider the type of artifact you are analyzing. For instance, if the artifact is a conversation, include points and questions that cover different perspectives and aspects discussed during the conversation.

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=document),
        ]
    )


def get_text(path):
    if not path:  # read from stdin
        return "".join(sys.stdin.readlines())
    # check if path is URL
    if path.startswith("http"):
        request = requests.get(path)
        out = html2text.html2text(request.text)
        return out
    if path:
        # try to open the file, using pathlib
        return Path(path).read_text()
    # read stdin
    return str(sys.stdin.readlines())


PRINT_BUFFER = ""


def println(s):
    global PRINT_BUFFER
    PRINT_BUFFER += s + "\n"


async def a_think(gist: bool, path: str):
    llms = [
        langchain_helper.get_model(openai=True),
        langchain_helper.get_model(claude=True),
        langchain_helper.get_model(google=True),
    ]

    user_text = get_text(path)
    ic("starting to think", num_tokens_from_string(user_text))
    if path:
        println(f"*Thinking about {path}*")
    println("* 🧠 via [think.py](https://github.com/idvorkin/nlp/blob/main/think.py).*")

    def do_llm_think(llm) -> List[[str, BaseChatModel]]:  # type: ignore
        from langchain.schema.output_parser import StrOutputParser

        # return prompt_think_about_document(user_text) | llm.with_structured_output( AnalyzeArtifact)
        return prompt_think_about_document(user_text) | llm | StrOutputParser()

    analyzed_artifacts = await langchain_helper.async_run_on_llms(do_llm_think, llms)

    for analysis, llm, duration in analyzed_artifacts:
        println(
            f"# -- model: {langchain_helper.get_model_name(llm)} | {duration.total_seconds():.2f} seconds --"
        )
        println(analysis)
    if gist:
        # create temp file and write print buffer to it
        output = Path("~/tmp/think.md")  # get smarter about naming these.
        output.write_text(PRINT_BUFFER)
        langchain_helper.to_gist(output)
    else:
        print(PRINT_BUFFER)


console = Console()
app = typer.Typer()


@app.command()
def think(
    trace: bool = False,
    gist: bool = False,
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_think(gist, path))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
