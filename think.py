#!python3


import re
from pathlib import Path
import asyncio
from typing import List
from datetime import datetime, timedelta
from langchain_core import messages
from langchain_core.language_models import BaseChatModel
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models import BaseChatModel

import typer
from langchain.prompts import ChatPromptTemplate

from loguru import logger
from rich.console import Console
import langchain_helper
import openai_wrapper
from icecream import ic
from openai_wrapper import num_tokens_from_string
from pydantic import BaseModel
from exa_py import Exa
import os
import requests
from bs4 import BeautifulSoup


class AnalysisResult(BaseModel):
    analysis: str
    llm: BaseChatModel 
    duration: timedelta

class AnalysisBody(BaseModel):
    body: str
    artifacts: List[AnalysisResult]

class CategoryInfo(BaseModel):
    categories: List[str]
    description: str

class GroupOfPoints(BaseModel):
    Description: str
    Points: List[str]


class Section(BaseModel):
    Title: str
    Topics: List[GroupOfPoints]


class ArtifactReport(BaseModel):
    Sections: List[Section]


class AnalysisQuestions:
    @staticmethod
    def default():
        return [
            "Summary",
            "Most Novel Ideas",
            "Most Interesting Ideas",
            "Critical Assumptions and Risks",
            "Reflection Questions",
            "Contextual Background",
            "Related Topics",
        ]

    @staticmethod
    def interests():
        return [
            "Summary",
            "Implications and Impact",
            "Most Novel Ideas" "Most Interesting Ideas" "Reflection Questions",
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

    @staticmethod
    def writer():
        return [
            "Who are possible audiences of this, and what will they find most important?"
            "What are 5 other topics we could develop?"
            "What would make this better?"
            "What are novel points and why?"
            "What could make this funnier?"
            "What are 5 alternative (include witty, funny, catchy) titles?"
        ]


def prompt_think_about_document(document, categories):
    description_of_point_form = """
### Title for Group:
 - Point 1
 - Point 2
 - ...
### Title for Group:
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

Please format your analysis as follows (**do not** title the groups as group, but use the name of the group), use markdown:

{example}

Ensure that you consider the type of artifact you are analyzing. For instance, if the artifact is a conversation, include points and questions that cover different perspectives and aspects discussed during the conversation.

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=document),
        ]
    )


def sanitize_filename(filename: str) -> str:
    """Convert a string into a safe filename."""
    # Replace invalid characters with underscores
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove any non-ASCII characters
    filename = "".join(char for char in filename if ord(char) < 128)
    return filename.strip()


def make_summary_prompt(content: str, sections: List[str]):
    # Create a summarization prompt for models to analyze all outputs
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(
                content=f"""You are an expert at synthesizing multiple analyses into clear, actionable insights.
    Review the analyses below from different AI models and create a concise summary that:
    1. Identifies the most valuable insights across all analyses
    2. Ranks points by importance and actionability
    3. Groups related ideas together
    4. Highlights where models agree and where only 1 model observes something
    4.1 Where models (or a subset of models) agree, include the point form summary of each agreement
    4.2 Where only a single model observes something, include the original text from the models that disagree
    5. Preserves the original section structure e.g.
    {sections}


    Format your response in markdown with:
    - Clear section headers
    - Bullet points for key insights
    - Brief notes on model consensus/disagreement where relevant
    """
            ),
            messages.HumanMessage(content=content),
        ]
    )


# Helper function for parallel summary generation
async def generate_model_summary(llm, summary_prompt, header, output_dir, duration):
    model_name = langchain_helper.get_model_name(llm)
    try:
        summary = await (summary_prompt | llm).ainvoke({})

        if not summary:  # Add error handling for empty summaries
            ic(f"Warning: Empty summary from {model_name}")
            return None

        summary_path = output_dir / f"summary_{sanitize_filename(model_name)}.md"
        summary_text = f"""
# Model Summary by {model_name}
{header}
Duration: {duration.total_seconds():.2f} seconds

{summary.content if hasattr(summary, 'content') else summary}
"""
        summary_path.write_text(summary_text)
        return summary_path
    except Exception as e:
        ic(f"Error generating summary for", model_name, e)
        return None



def get_categories_and_description(core_problems: bool, writer: bool, interests: bool) -> CategoryInfo:
    categories = AnalysisQuestions.default()
    category_desc = "default questions"
    
    if core_problems:
        categories = AnalysisQuestions.core_problem()
        category_desc = "core problems"
    if writer:
        categories = AnalysisQuestions.writer()
        category_desc = "writer questions"
    if interests:
        categories = AnalysisQuestions.interests()
        category_desc = "interests"
        
    return CategoryInfo(categories=categories, description=category_desc)

async def generate_analysis_body(user_text: str, categories: List[str], llms: List[BaseChatModel]) -> AnalysisBody:
    def do_llm_think(llm):
        return (
            prompt_think_about_document(user_text, categories=categories)
            | llm
            | StrOutputParser()
        )

    analyzed_artifacts = await langchain_helper.async_run_on_llms(do_llm_think, llms)
    
    results = [
        AnalysisResult(analysis=analysis, llm=llm, duration=duration)
        for analysis, llm, duration in analyzed_artifacts
    ]

    body = ""
    for result in results:
        body += f"""
<details>
<summary>

# -- model: {langchain_helper.get_model_name(result.llm)} | {result.duration.total_seconds():.2f} seconds --

</summary>

{result.analysis}

</details>

"""
    return AnalysisBody(body=body, artifacts=results)

async def a_think(
    gist: bool, writer: bool, path: str, core_problems: bool, interests: bool
):
    output_dir = Path("~/tmp").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    llms = langchain_helper.get_models(openai=True, claude=True, google=True)

    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    tokens = num_tokens_from_string(user_text)

    if tokens < 16_000:
        llms += [langchain_helper.get_model(llama=True)]

    category_info = get_categories_and_description(core_problems, writer, interests)
    
    title = ""
    if path and path.startswith(("http://", "https://")):
        try:
            response = requests.get(path, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            title = f" ({soup.title.string.strip('()')})" if soup.title else ""
        except Exception as _:
            pass

    thinking_about = (
        f"*Thinking about [{title}]({path})*"
        if title
        else f"*Thinking about [{path}]({path})*"
        if path
        else ""
    )

    today = datetime.now().strftime("%Y-%m-%d")
    header = f"""
*🧠 via [think.py](https://github.com/idvorkin/nlp/blob/main/think.py) - {today} - using {category_info.description}* <br/>
{thinking_about}
"""

    ic("starting to think", tokens)
    analysis_body = await generate_analysis_body(user_text, category_info.categories, llms)
    output_text = header + "\n" + analysis_body.body

    # Create the main analysis file
    output_path = output_dir / "think.md"
    output_path.write_text(output_text)

    # Run all model summaries in parallel
    model_summary_tasks = [
        generate_model_summary(
            result.llm,
            make_summary_prompt(analysis_body.body, category_info.categories),
            header,
            output_dir,
            result.duration
        )
        for result in analysis_body.artifacts
    ]
    model_summaries = [
        summary
        for summary in await asyncio.gather(*model_summary_tasks)
        if summary is not None
    ]

    # Create list of files to include in gist
    files_to_gist = [output_path] + model_summaries

    if gist:
        # Use to_gist_multiple instead of to_gist
        langchain_helper.to_gist_multiple(files_to_gist)
    else:
        print(output_text)
        for summary_path in model_summaries:
            print(f"\n=== Summary by {summary_path.stem} ===\n")
            print(summary_path.read_text())


console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def think(
    trace: bool = False,
    gist: bool = True,
    core_problems: bool = False,  # Use core problems answers
    writer: bool = False,  # Use core problems answers
    interests: bool = False,  # Use core problems answers
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace,
        lambda: asyncio.run(
            a_think(
                gist=gist,
                writer=writer,
                path=path,
                core_problems=core_problems,
                interests=interests,
            )
        ),
    )


@logger.catch()
def app_wrap_loguru():
    app()


def exa_search(query: str, num_results: int = 20) -> str:
    exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

    if not isinstance(query, str) or not query.startswith(("http://", "https://")):
        return ""

    results = exa.find_similar_and_contents(
        query,
        num_results=num_results,
        summary=True,
        highlights={"num_sentance": 3, "highlights_per_url": 2},
    )

    search_results = ""
    for result in results.results:
        search_results += f"- [{result.title}]({result.url})\n"
        search_results += f"  - {result.summary}\n"
        for highlight in result.highlights:
            search_results += f"      - {highlight}\n"

    return search_results


if __name__ == "__main__":
    app_wrap_loguru()
