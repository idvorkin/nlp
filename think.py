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
import openai_wrapper
from icecream import ic
from openai_wrapper import num_tokens_from_string
from pydantic import BaseModel


class GroupOfPoints(BaseModel):
    Description: str
    Points: List[str]


class Section(BaseModel):
    Title: str
    Topics: List[GroupOfPoints]


class ArtifactReport(BaseModel):
    Sections: List[Section]


def markdown_to_analyze_artifact(markdown: str) -> ArtifactReport:
    sections = []
    current_section: Section = None  # type: ignore
    current_topic: GroupOfPoints = None  # type: ignore

    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("## "):  # This is a section
            if current_section:
                sections.append(current_section)
            current_section = Section(Title=line[3:].strip(), Topics=[])
        elif line.startswith("### "):  # This is a topic
            if current_topic:
                current_section.Topics.append(current_topic)
            current_topic = GroupOfPoints(Description=line[4:].strip(), Points=[])
        elif line.startswith("* ") or line.startswith("- "):  # This is a point
            if current_topic is not None:
                current_topic.Points.append(line[2:].strip())

    # Add the last topic and section
    if current_topic:
        current_section.Topics.append(current_topic)
    if current_section:
        sections.append(current_section)

    return ArtifactReport(Sections=sections)


def merge_analyze_artifacts(reports: List[ArtifactReport]) -> ArtifactReport:
    all_sections = [section for report in reports for section in report.Sections]
    unique_section_titles = list(set([section.Title for section in all_sections]))

    merged_report = ArtifactReport(Sections=[])

    for section_title in unique_section_titles:
        # get the topics for that section across all artifacts
        sections = [
            section
            for report in reports
            for section in report.Sections
            if section.Title == section_title
        ]
        ic(section_title, sections)
        # flatten the topics
        topics = [topic for section in sections for topic in section.Topics]
        ic(topics)

        merged_report.Sections.append(Section(Title=section_title, Topics=topics))

    return merged_report


class AnalysisQuestions:
    @staticmethod
    def default():
        return [
            "Summary",
            "Implications and Impact",
            "Most Novel Ideas"
            "Most Interesting Ideas"
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


async def a_think(gist: bool, writer: bool, path: str, core_problems: bool):
    # claude is now too slow to use compared to gpto
    llms = langchain_helper.get_models(openai=True, claude=True)

    user_text = openai_wrapper.get_text_from_path_or_stdin(path)
    tokens = num_tokens_from_string(user_text)

    if tokens < 16_000:  # Groq limits to 60K ish
        # only add Llama if the text is small
        llms += [langchain_helper.get_model(llama=True)]

    categories = AnalysisQuestions.default()
    category_desc = "default questions"
    if core_problems:
        categories = AnalysisQuestions.core_problem()
        category_desc = "core problems"
    if writer:
        categories = AnalysisQuestions.writer()
        category_desc = "writer questions"

    # todo add link to categories being used.

    thinking_about = f"*Thinking about {path}*" if path else ""
    ic("starting to think", tokens)
    header = f"""
*🧠 via [think.py](https://github.com/idvorkin/nlp/blob/main/think.py) - using {category_desc}*
{thinking_about}
    """

    def do_llm_think(llm) -> List[[str, BaseChatModel]]:  # type: ignore
        from langchain.schema.output_parser import StrOutputParser

        # return prompt_think_about_document(user_text) | llm.with_structured_output( AnalyzeArtifact)
        return (
            prompt_think_about_document(user_text, categories=categories)
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
    # parsed = [markdown_to_analyze_artifact(analysis) for analysis, _, _ in analyzed_artifacts]
    # merged = merge_analyze_artifacts(parsed)
    # # overwrite the body text with the parsed version
    # body += f"""
    # --Merged--
    # {merged.model_dump_json(indent=2)}
    # """

    output_text = header + "\n" + body
    output_path = Path("~/tmp/think.md").expanduser()  # get smarter about naming these.
    output_path.write_text(output_text)
    ic(output_path)
    if gist:
        # create temp file and write print buffer to it
        langchain_helper.to_gist(output_path)
    else:
        print(output_text)


console = Console()
app = typer.Typer(no_args_is_help=True)


@app.command()
def think(
    trace: bool = False,
    gist: bool = True,
    core_problems: bool = False,  # Use core problems answers
    writer: bool = False,  # Use core problems answers
    path: str = typer.Argument(None),
):
    langchain_helper.langsmith_trace_if_requested(
        trace,
        lambda: asyncio.run(
            a_think(gist=gist, writer=writer, path=path, core_problems=core_problems)
        ),
    )


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
