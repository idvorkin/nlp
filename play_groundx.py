#!python3


import typer

from loguru import logger
from rich.console import Console
from icecream import ic
from groundx import Groundx
import os

GROUNDX_API_KEY = os.getenv("GROUNDX_API_KEY")
IGOR_BLOG_BUCKET = 8545
groundx = Groundx(api_key=GROUNDX_API_KEY)

console = Console()
app = typer.Typer(no_args_is_help=True)


@logger.catch()
def app_wrap_loguru():
    app()


@app.command()
def ask(question: str):
    result = groundx.search.content(
        id=IGOR_BLOG_BUCKET,
        n=10,
        query=question,
    )
    llmText = result.body["search"]["text"]
    print(llmText)


@app.command()
def load_blog():
    response = groundx.documents.crawl_website(
        websites=[
            {
                "bucketId": IGOR_BLOG_BUCKET,
                "cap": 500,
                "depth": 5,
                "searchData": {"key": "value"},
                # "sourceUrl": "https://idvork.in/all.html"
                "sourceUrl": "https://idvork.in/manager-book",
            }
        ]
    )
    ic(response)


if __name__ == "__main__":
    app_wrap_loguru()
