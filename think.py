#!python3


import re
from pathlib import Path
import asyncio
from typing import List
from datetime import datetime, timedelta
from langchain_core import messages
from github_helper import get_latest_github_commit_url, get_repo_info
from typing import List
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




class ModelTiming(BaseModel):
    analysis_duration: timedelta
    summary_duration: timedelta

class AnalysisResult(BaseModel):
    analysis: str
    llm: BaseChatModel
    duration: timedelta
    summary_duration: timedelta | None = None
    summary_content: str = ""  # Store the actual summary content

class AnalysisBody(BaseModel):
    body: str
    artifacts: List[AnalysisResult]
    total_analysis_time: timedelta
    total_summary_time: timedelta
    exa_results: str = ""

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
async def generate_model_summary(llm, summary_prompt, header, output_dir, analysis_duration):
    model_name = langchain_helper.get_model_name(llm)
    start_time = datetime.now()
    try:
        summary = await (summary_prompt | llm).ainvoke({})
        end_time = datetime.now()
        summary_duration = end_time - start_time

        if not summary:
            ic(f"Warning: Empty summary from {model_name}")
            return None

        # Clean up the summary content
        summary_content = summary.content if hasattr(summary, 'content') else summary
        
        # Handle case where summary is a list (from Gemini)
        if isinstance(summary_content, list):
            # Take the second element which typically contains the actual summary
            summary_content = summary_content[1] if len(summary_content) > 1 else summary_content[0]

        # Remove markdown code block tags and thinking process
        summary_content = re.sub(r'```markdown\n|\n```', '', summary_content)
        if '["' in summary_content:  # Check for thinking process
            # Extract just the markdown content after the thinking process
            summary_content = summary_content.split('```markdown\n')[-1].split('\n```')[0]

        summary_path = output_dir / f"summary_{sanitize_filename(model_name)}.md"
        summary_text = f"""# Model Summary by {model_name}
{header}
Analysis Duration: {analysis_duration.total_seconds():.2f} seconds
Summary Duration: {summary_duration.total_seconds():.2f} seconds

{summary_content}
"""
        summary_path.write_text(summary_text)
        return summary_path, summary_duration, summary_content  # Return the summary content
    except Exception as e:
        ic(f"Error generating summary for {model_name}: {e}")
        error_content = f"Error generating summary for {model_name}: {e}"
        summary_path = output_dir / f"summary_{sanitize_filename(model_name)}_error.md"
        summary_text = f"""# Model Summary by {model_name}
{header}
Analysis Duration: {analysis_duration.total_seconds():.2f} seconds
Summary Duration: N/A

{error_content}
"""
        summary_path.write_text(summary_text)
        return summary_path, None, error_content



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
        category_desc = "interests questions"

    return CategoryInfo(categories=categories, description=category_desc)

async def generate_analysis_body(user_text: str, categories: List[str], llms: List[BaseChatModel], path: str = "") -> AnalysisBody:
    def do_llm_think(llm):
        return (
            prompt_think_about_document(user_text, categories=categories)
            | llm
            | StrOutputParser()
        )

    analyzed_artifacts = []
    for llm in llms:
        try:
            result = await langchain_helper.async_run_on_llms(do_llm_think, [llm])
            analyzed_artifacts.append(result[0])  # Unpack the single result
        except Exception as e:
            ic(f"Error analyzing with {langchain_helper.get_model_name(llm)}: {e}")
            analyzed_artifacts.append((f"Error analyzing with {langchain_helper.get_model_name(llm)}: {e}", llm, timedelta()))

    results = [
        AnalysisResult(analysis=analysis, llm=llm, duration=duration)
        for analysis, llm, duration in analyzed_artifacts
    ]

    # Calculate total analysis time
    total_analysis_time = sum((result.duration for result in results), timedelta())

    body = ""
    for result in results:
        body += f"""
<details>
<summary>

# -- {langchain_helper.get_model_name(result.llm)} | {result.duration.total_seconds():.2f} seconds --

</summary>

{result.analysis}

</details>

"""

    # Add Exa results if path exists
    exa_content = ""
    if path:
        try:
            exa_content = exa_search(path)
        except Exception as e:
            ic(f"Error fetching Exa results: {e}")
            exa_content = f"Error fetching Exa results: {e}"
        if exa_content:
            body += f"""
<details>
<summary>

# -- Related Content (via Exa) --

</summary>

{exa_content}

</details>

"""

    return AnalysisBody(
        body=body, 
        artifacts=results,
        total_analysis_time=total_analysis_time,
        total_summary_time=timedelta(),  # Initialize with zero, will be updated later
        exa_results=exa_content
    )

def create_overview_content(header: str, analysis_body: AnalysisBody, model_summaries: List[Path]) -> str:
    # Start with the header
    overview = f"{header}\n\n"
    
    # Add analysis files link
    overview += "- [Complete Analysis](#file-b_think-md)\n"
    
    # Add timing breakdown table without a header
    overview += "\n| Model | Analysis (seconds) | Summary (seconds) | Analysis Size (KB) | Summary Size (KB) |\n"
    overview += "|-------|-------------------|------------------|------------------|------------------|\n"
    
    # Sort by model name
    sorted_results = sorted(analysis_body.artifacts, 
                          key=lambda x: langchain_helper.get_model_name(x.llm).lower())

    # Initialize totals
    total_analysis_size = 0
    total_summary_size = 0
    
    for result in sorted_results:
        model_name = langchain_helper.get_model_name(result.llm)
        safe_name = sanitize_filename(model_name).lower().replace('.', '-')
        model_link = f"[{model_name}](#file-summary_{safe_name}-md)"
        
        # Calculate sizes in KB
        analysis_size = len(result.analysis) / 1024
        summary_size = len(result.summary_content) / 1024 if result.summary_content else 0
        
        # Add to totals
        total_analysis_size += analysis_size
        total_summary_size += summary_size
        
        # Format durations and sizes
        analysis_duration = f"{result.duration.total_seconds():.2f}"
        summary_duration = f"{result.summary_duration.total_seconds():.2f}" if result.summary_duration else "N/A"
        analysis_kb = f"{analysis_size:.1f}"
        summary_kb = f"{summary_size:.1f}"
        
        overview += f"| {model_link} | {analysis_duration} | {summary_duration} | {analysis_kb} | {summary_kb} |\n"
    
    # Add totals row without separator
    overview += f"| **Total** | | | **{total_analysis_size:.1f}** | **{total_summary_size:.1f}** |\n"
    
    if analysis_body.exa_results:
        overview += "\n| Source | Content |\n"
        overview += "|--------|----------|\n"
        overview += "| Exa Search | See [Complete Analysis](#file-b_think-md) |\n"
    
    return overview

async def a_think(
    gist: bool, writer: bool, path: str, core_problems: bool, interests: bool
):
    output_dir = Path("~/tmp").expanduser()
    repo_info = get_repo_info()  # Default False for getting source file URL
    output_dir.mkdir(parents=True, exist_ok=True)
    llms = langchain_helper.get_models(openai=True, claude=True, google=True, google_think=True, deepseek=True, o4_mini=True, google_flash=True, openai_mini=True)

    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    tokens = num_tokens_from_string(user_text)

    if tokens < 30_000:
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
*🧠 via [think.py]({get_latest_github_commit_url(repo_info.name, "think.py")}) - {today} - using {category_info.description}* <br/>
{thinking_about}
"""

    ic("starting to think", tokens)
    analysis_body = await generate_analysis_body(user_text, category_info.categories, llms, path)
    output_text = header + "\n" + analysis_body.body

    # Create the main analysis file
    output_path = output_dir / "b_think.md"
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
    summary_results = [
        summary
        for summary in await asyncio.gather(*model_summary_tasks)
        if summary is not None
    ]
    
    # Unpack the results - now including summary content
    model_summaries = [path for path, _, _ in summary_results]
    summary_durations = [duration for _, duration, _ in summary_results]
    
    # Update analysis results with summary content
    for result, (_, duration, content) in zip(analysis_body.artifacts, summary_results):
        result.summary_duration = duration
        result.summary_content = content

    # Calculate total summary time and update analysis results
    total_summary_time = sum((duration for _, duration, _ in summary_results), timedelta())
    
    # Update the analysis results with summary durations
    for result, (_, duration, _) in zip(analysis_body.artifacts, summary_results):
        result.summary_duration = duration
    
    analysis_body.total_analysis_time = sum((r.duration for r in analysis_body.artifacts), timedelta())
    analysis_body.total_summary_time = total_summary_time

    # Create overview file with actual timings
    # Get title from the soup if available
    overview_filename = "a_overview"
    if title:
        # Clean the title by removing parentheses and extra spaces
        clean_title = title.strip('() ')
        # Convert to filename-safe format
        clean_title = sanitize_filename(clean_title).lower().replace(' ', '-')
        overview_filename = f"a_{clean_title}--overview"
    
    overview_path = output_dir / f"{overview_filename}.md"
    overview_content = create_overview_content(header, analysis_body, model_summaries)
    overview_path.write_text(overview_content)

    # Create list of files to include in gist, with overview first
    files_to_gist = [overview_path, output_path] + model_summaries

    if gist:
        # Get the title to use in gist description
        gist_description = f"think - {title.strip('() ')}" if title else "think"
        # Clean up description by removing any newlines and truncating if too long
        gist_description = gist_description.replace('\n', ' ')[:100]
        # Use to_gist_multiple with description
        langchain_helper.to_gist_multiple(files_to_gist, description=gist_description)
    else:
        print(overview_content)
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

    try:
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
    except Exception as e:
        ic(f"Error during Exa search: {e}")
        return f"Error during Exa search: {e}"


if __name__ == "__main__":
    app_wrap_loguru()
