#!python3

import sys
import typer
from loguru import logger
from rich.console import Console
from icecream import ic
import openai_wrapper
import ell
from datetime import datetime
from ell_helper import init_ell, run_studio, get_ell_model
from typer import Option

console = Console()
app = typer.Typer(no_args_is_help=True)

# Initialize ELL
init_ell()


@logger.catch()
def app_wrap_loguru():
    app()


@ell.simple(model=get_ell_model(claude=True), max_tokens=8000)
def prompt_podcast_for_book(book: str):
    """
    You are an AI assistant tasked with creating an engaging podcast-style conversation between two hosts discussing a book. The book to be discussed is provided to you


    Your task is to generate a JSON object containing two main components: podcast preparation and the conversation itself. Follow these steps:

    1. Podcast Preparation:
       Use <podcast_preparation> tags to organize your thoughts and plan the podcast structure. Include the following elements:
       a) Book summary: Provide a concise summary of the book's plot, main characters, and central themes. Explain your thought process in identifying the key elements.
       b) Character analysis: List the main characters and briefly describe their roles and development. Explain why you chose these characters as significant.
       c) Theme exploration: Identify and explain the book's major themes. Describe how you recognized these themes and their importance to the story.
       d) Discussion points: Create 3-5 interesting discussion points or questions. Explain why you think each point would be engaging for listeners.
       e) Conversation outline: Develop a rough outline of the conversation structure (intro, main discussion, conclusion). Explain your rationale for this structure.
       f) Casual expressions and jokes: Generate some casual expressions or jokes related to the book. Explain how these might enhance the podcast's entertainment value.
       g) Potential listener questions: Brainstorm 2-3 questions that listeners might have, and briefly note how the hosts could address them.
       h) Target audience consideration: Reflect on how to tailor the conversation to appeal to the podcast's target audience.

       Remember to keep the tone casual and entertaining throughout the planning process, as if you're chatting with a friend about the book.

    2. Conversation Generation:
       Create a fun and light podcast-style conversation between two hosts (H1 and H2) discussing the book. The conversation should:
       a) Include casual banter and natural dialogue exchanges
       b) Maintain an engaging and entertaining tone
       c) Cover key aspects of the book without being overly formal
       d) Use colloquial language and expressions where appropriate
       e) Flow naturally with questions, responses, and occasional interjections
       f) Address potential listener questions identified in the preparation phase
       g) Tailor content to appeal to the target audience

    Format the conversation as a series of alternating speaker lines, using "H1" for Host 1 and "H2" for Host 2. Begin with a brief introduction and end with a conclusion or sign-off.

    Output the result as a JSON object with the following structure:

    {
      "podcast_preparation": {
        "summary": "Book summary here",
        "characters": ["Character 1: description", "Character 2: description"],
        "themes": ["Theme 1: explanation", "Theme 2: explanation"],
        "discussion_points": ["Point 1", "Point 2", "Point 3"],
        "outline": ["Intro", "Main discussion", "Conclusion"],
        "casual_expressions": ["Expression 1", "Expression 2"],
        "listener_questions": ["Question 1", "Question 2"],
        "audience_tailoring": "Notes on appealing to target audience"
      },
      "conversation": [
        {"speaker": "H1", "text": "Introduction dialogue"},
        {"speaker": "H2", "text": "Response dialogue"},
        // ... more dialogue entries
        {"speaker": "H1", "text": "Conclusion dialogue"}
      ]
    }

    Ensure that your JSON is properly formatted and valid. Begin your response with the podcast preparation phase, then proceed to generate the conversation based on your planning.
    """
    return book


@app.command()
def for_book(
    studio: bool = Option(False, help="Launch the ELL Studio interface"),
    port: int = Option(
        None, help="Port to run the ELL Studio on (only used with --studio)"
    ),
):
    """
    Create a podcast for the book
    """
    if studio:
        run_studio(port=port)
        return

    start = datetime.now()

    user_text = "".join(sys.stdin.readlines())
    fixed = prompt_podcast_for_book(user_text)
    elapsed_seconds = (datetime.now() - start).total_seconds()
    ic(openai_wrapper.num_tokens_from_string(user_text))
    ic(elapsed_seconds)
    print(fixed)


if __name__ == "__main__":
    app_wrap_loguru()
