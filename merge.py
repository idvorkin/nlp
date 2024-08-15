#!python3


from pathlib import Path
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


def prompt_patch_in_document(source_path:Path, base_artifcat, to_merge):
    instructions = f"""
You are a brilliant writer, and help users merge content into their blog posts. Below is the base document

The name of the base document is:  {source_path.name}

<base_document>
{base_artifcat}
</base_document>

Create a diff that can be applied via patch to merge the new content into the base document from the content provided by the user. Only output the patch file. Do not include anything else (including back ticks)

Be intelligent in the merge, adding the content to the correct locations

"""
    return ChatPromptTemplate.from_messages(
        [
            messages.SystemMessage(content=instructions),
            messages.HumanMessage(content=to_merge),
        ]
    )



console = Console()
app = typer.Typer()


@app.command()
def merge(
    source: Path = typer.Argument(help="Source File"), # Source File
    merge_path: Path = typer.Argument(None, help="content to merge, path or stdin"), # File to merge
):

    from langchain.schema.output_parser import StrOutputParser
    llm = langchain_helper.get_model(openai=True)
    merge_text = langchain_helper.get_text_from_path_or_stdin(merge_path)
    prompt = prompt_patch_in_document(source, Path(source).read_text(), merge_text)
    patch_chain =  prompt | llm | StrOutputParser()

    patch = patch_chain.invoke({})
    patch_file_path="merge.patch"
    Path(patch_file_path).write_text(patch)
    print(patch)
    ic(patch_file_path)

    """ Create a diff of merge_path to source"""
    return




@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
