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


class AnalysisQuestions:
    @staticmethod
    def default():
        return [
            "Summary",
            "Implications and Impact",
            "Critical Assumptions and Risks",
            "Reflection Questions",
            "Contextual Background",
            "Related Topics",
        ]

    @staticmethod
    def core_problem():
        return [
            "What's the real problem you are trying to solve?",
            "What's your hypothesis? Why?",
            "What are your core assumptions? Why?",
            "What evidence do you have?",
            "What are your core options?",
            "What alternatives exist?",
        ]


def prompt_think_about_document(document, categories):
    description_of_point_form = """
### Group:
 - Point 1
 - Point 2
 - ...
### Group:
 - Point 1
 - Point 2
 - ...
   - ...
    """

    # have first 2 include the summary
    example = ""
    for i, category in enumerate(categories):
        example += f"## {category}\n\n"
        if i < 2:
            example += description_of_point_form
        else:  # just one group
            example += "\n [as above] \n"

    instructions = f"""
You are a brilliant expert at critical thinking, specialized in digesting and enhancing understanding of various artifacts. The user will rely on you to help them think critically about the thing they are reading.

For this task, you will analyze the provided artifact. Your aim is to structure your analysis into the sections listed below.  Each section should contain between 2 and 5 groups of points. Each group should include 2 to 10 specific points that are critical to understanding the artifact.

Please format your analysis as follows (do not use the word group, but use the actual group or topic), use markdown:

{example}

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


async def a_think(gist: bool, path: str, core_problems: bool):
    llms = [
        langchain_helper.get_model(openai=True),
        langchain_helper.get_model(claude=True),
        # langchain_helper.get_model(google=True),
    ]

    user_text = get_text(path)
    tokens = num_tokens_from_string(user_text)

    categories = AnalysisQuestions.default()
    category_desc = "default questions"
    if core_problems:
        categories = AnalysisQuestions.core_problem()
        category_desc = "core problems"

    if tokens < 8000:
        # only add Llama if the text is small
        llms += [langchain_helper.get_model(llama=True)]

    # todo add link to categories being used.
    ic("starting to think", tokens)
    if path:
        println(f"*Thinking about {path}*")
    println(
        f"*🧠 via [think.py](https://github.com/idvorkin/nlp/blob/main/think.py) - using {category_desc}*"
    )

    def do_llm_think(llm) -> List[[str, BaseChatModel]]:  # type: ignore
        from langchain.schema.output_parser import StrOutputParser

        # return prompt_think_about_document(user_text) | llm.with_structured_output( AnalyzeArtifact)
        return (
            prompt_think_about_document(user_text, categories=categories)
            | llm
            | StrOutputParser()
        )

    analyzed_artifacts = await langchain_helper.async_run_on_llms(do_llm_think, llms)

    for analysis, llm, duration in analyzed_artifacts:
        println(
            f"# -- model: {langchain_helper.get_model_name(llm)} | {duration.total_seconds():.2f} seconds --"
        )
        println(analysis)

    output_path = Path("~/tmp/think.md").expanduser()  # get smarter about naming these.
    output_path.write_text(PRINT_BUFFER)
    ic(output_path.name)
    if gist:
        # create temp file and write print buffer to it
        langchain_helper.to_gist(output_path)
    else:
        print(PRINT_BUFFER)


console = Console()
app = typer.Typer()


@app.command()
def think(
    trace: bool = False,
    gist: bool = False,
    core_problems: bool = False,  # Use core problems answers
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace, lambda: asyncio.run(a_think(gist, path, core_problems=core_problems))
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
