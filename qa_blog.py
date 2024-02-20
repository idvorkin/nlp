#!python3
import requests
from functools import lru_cache
import pathlib
from langchain_openai.chat_models import ChatOpenAI
from rich.console import Console
from icecream import ic
import typer
import os
from rich import print
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import (
    text_splitter,
)  # import CharacterTextSplitter, RecursiveCharacterTextSplitter, Markdown
from typing_extensions import Annotated
from openai_wrapper import choose_model
from fastapi import FastAPI
from openai_wrapper import setup_gpt, get_model
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
import json
import discord
from discord_helper import draw_progress_bar, get_bot_token, send


gpt_model = setup_gpt()
server = FastAPI()

app = typer.Typer()
console = Console()

bot = discord.Bot()
chroma_db_dir = "blog.chroma.db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = OpenAIEmbeddings()

g_debug_out = ""


chunk_size_5k_tokens = (
    4 * 1000 * 5
)  # ~ 5K tokens, given we'll be doing 5-10 facts, seems reasonable


def chunk_documents_recursive(documents, chunk_size=chunk_size_5k_tokens):
    recursive_splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_size // 4
    )
    splitter = recursive_splitter

    for document in documents:
        chunks = splitter.split_text(document.page_content)
        for chunk in chunks:
            d = Document(
                page_content=chunk,
                metadata={
                    "chunk_method": "recursive_char",
                    "source": document.metadata["source"],
                    "is_entire_document": len(chunks) == 1,
                },
            )
            ic(d.metadata)
            yield d


def chunk_documents_as_md(documents, chunk_size=chunk_size_5k_tokens):
    # TODO: Use UnstructuredMarkdownParser
    # Interesting trade off here, if we make chunks bigger we can have more context
    # If we make chunk smaller we can inject more chunks
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]
    markdown_splitter = text_splitter.MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    splitter = markdown_splitter

    for document in documents:
        base_metadata = {
            "source": document.metadata["source"],
            "chunk_method": "md_simple",
            "is_entire_document": False,
        }
        for chunk in splitter.split_text(document.page_content):
            yield Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, **base_metadata},
            )


def chunk_documents_as_md_large(documents, chunk_size=chunk_size_5k_tokens):
    # TODO: Use UnstructuredMarkdownParser
    # Interesting trade off here, if we make chunks bigger we can have more context
    # If we make chunk smaller we can inject more chunks
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]
    markdown_splitter = text_splitter.MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    splitter = markdown_splitter

    for document in documents:
        base_metadata = {
            "source": document.metadata["source"],
            "chunk_method": "md_merge",
        }
        candidate_chunk = Document(page_content="", metadata=base_metadata)
        is_entire_document = True
        for chunk in splitter.split_text(document.page_content):
            candidate_big_enough = len(candidate_chunk.page_content) > chunk_size
            if candidate_big_enough:
                is_entire_document = False
                candidate_chunk.metadata["is_entire_document"] = is_entire_document
                yield candidate_chunk
                candidate_chunk = Document(page_content="", metadata=base_metadata)

            # grow the candate chunk with current chunk
            candidate_chunk.page_content += chunk.page_content

        # yield the last chunk, regardless of its size
        candidate_chunk.metadata["is_entire_document"] = is_entire_document
        yield candidate_chunk


def get_blog_content(path):
    # set_trace()
    repo_path = pathlib.Path(os.path.expanduser(path))

    markdown_files = list(repo_path.glob("*/*.md"))
    for markdown_file in markdown_files:
        with open(markdown_file, "r") as f:
            yield Document(
                page_content=f.read(),
                metadata={"source": str(markdown_file.relative_to(repo_path))},
            )


def dedup_chunks(chunks):
    # chunks is a list of documents created by multiple chunkers
    # if we have multiple chunks from the same source and that contain the full document
    # only keep the first one
    unique_chunks = []
    seen_full_size = set()
    for chunk in chunks:
        source = chunk.metadata["source"]
        whole_doc = chunk.metadata["is_entire_document"]
        if whole_doc and source in seen_full_size:
            continue
        if whole_doc:
            seen_full_size.add(source)
        unique_chunks.append(chunk)
    return unique_chunks


@app.command()
def build():
    docs = list(get_blog_content("~/blog"))

    # It's OK, start by erasing the db
    # db_path = pathlib.Path(chroma_db_dir)
    # db_path.rmdir()

    ic(len(docs))
    chunks = list(chunk_documents_as_md(docs))
    chunks += list(chunk_documents_as_md_large(docs))
    chunks += list(chunk_documents_recursive(docs))
    deduped_chunks = dedup_chunks(chunks)
    ic(len(chunks), len(deduped_chunks))

    search_index = Chroma.from_documents(
        deduped_chunks, embeddings, persist_directory=chroma_db_dir
    )
    search_index.persist()


@app.command()
def chunk_md(
    path: Annotated[str, typer.Argument()] = "~/blog/_posts/2020-04-01-Igor-Eulogy.md",
):
    from unstructured.partition.md import partition_md

    elements = partition_md(filename=os.path.expanduser(path))
    ic(elements)


def fixup_markdown_path(src):
    # We built the file_path from source markdown
    def fixup_markdown_path_to_url(src):
        markdown_to_url = build_markdown_to_url_map()
        for md_file_path, url in markdown_to_url.items():
            # url starts with a /
            url = url[1:]
            md_link = f"[{url}](https://idvork.in/{url})"
            src = src.replace(md_file_path, md_link)
        return src

    def fixup_ig66_path_to_url(src):
        for i in range(100 * 52):
            src = src.replace(
                f"_ig66/{i}.md", f"[Family Journal {i}](https://idvork.in/ig66/{i})"
            )
        return src

    return fixup_ig66_path_to_url(fixup_markdown_path_to_url(src))


def has_whole_document(path):
    blog_content_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embeddings
    )
    all = blog_content_db.get()
    for m in all["metadatas"]:
        if m["source"] == path and m["is_entire_document"]:
            return True
    return False


def get_document(path):
    blog_content_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embeddings
    )
    all = blog_content_db.get()
    for i, m in enumerate(all["metadatas"]):
        if m["source"] == path and m["is_entire_document"]:
            return Document(page_content=all["documents"][i], metadata=m)
    raise Exception(f"{path} document found")


# cache this so it's memoized
@lru_cache
def build_markdown_to_url_map():
    source_file_to_url = {}
    # read the json file From Github, slightly stale, but good enough
    backlinks_url = "https://raw.githubusercontent.com/idvorkin/idvorkin.github.io/master/back-links.json"
    d = requests.get(backlinks_url).json()
    url_infos = d["url_info"]
    # "url_info": {
    # "/40yo": {
    # "markdown_path": "_d/40-yo-programmer.md",
    # "doc_size": 14000
    # },
    # convert the url_infos into a source_file_to_url map
    source_file_to_url = {v["markdown_path"]: k for k, v in url_infos.items()}
    return source_file_to_url


def docs_to_prompt(docs):
    ic(len(docs))
    ret = []
    for d in docs:
        d.metadata["source"] = fixup_markdown_path(d.metadata["source"])
        ret.append({"content": d.page_content, "metadata": d.metadata})

    return json.dumps(ret)
    # return "\n\n".join(doc.page_content for doc in docs)


@app.command()
def ask(
    question: Annotated[
        str, typer.Argument()
    ] = "What are the roles from Igor's Eulogy, answer in bullet form",
    facts: Annotated[int, typer.Option()] = 5,
    u4: bool = typer.Option(True),
    debug: bool = typer.Option(True),
):
    response = iask(question, facts, u4, debug)
    print(response)


async def iask(
    question: str,
    facts: int,
    u4: bool = True,
    debug: bool = True,
):
    model, _ = choose_model(u4)
    if debug:
        ic(model)
        ic(facts)
    # load chroma from DB
    blog_content_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embeddings
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
The content is all from Igor's blog
If you don't know the answer, just say that you don't know. Keep the answer under 10 lines

# The User's Questions
{question}

# Context
{context}

# Instruction

* Your answer should include sources like those listed below. The source files are markdown so if the have a header make an HTML anchor link when you make the source link. E.g. if it's in idvork.in, with header  # Foo , set it to http://idvork.in#foo


### Sources

* source file link  - Your reasoning on why it's  relevant (% relevance,  e.g. 20%)
    """
    )

    model_name = get_model(u4=True)
    llm = ChatOpenAI(model=model_name)
    # simple_retriever = blog_content_db.as_retriever(search_kwargs={"k": facts})

    # We can improve our relevance by getting the md_simple_chunks, but that loses context
    # Rebuild context by pulling in the largest chunk i can that contains the smaller chunk

    docs_and_scores = blog_content_db.similarity_search_with_relevance_scores(
        question, k=4 * facts
    )
    for doc, score in docs_and_scores:
        ic(doc.metadata, score)
    included_facts = [d for d, _ in docs_and_scores[:facts]]

    # included_facts = simple_retriever.get_relevant_documents(question)
    # included_facts = simple_retriever.get_relevant_documents(question)

    included_facts += [get_document("_posts/2020-04-01-Igor-Eulogy.md")]
    included_facts += [get_document("_d/operating-manual-2.md")]

    print("Source Documents")
    for doc in included_facts:
        # Remap metadata to url
        ic(doc.metadata)

    chain = prompt | llm | StrOutputParser()

    response = chain.ainvoke(
        {"question": question, "context": docs_to_prompt(included_facts)}
    )

    return await response


bot_help_text = "Replaced on_ready"


@bot.event
async def on_message(ctx):
    # if message is from me, skip it
    if ctx.author.bot:
        # ic ("Ignoring message from bot", message)
        return

    ic("bot.on_message", ctx)
    if len(ctx.content) == 0:
        return
    # message_content = ctx.content.replace(f"<@{bot.user.id}>", "").strip()
    await send(
        ctx, "Sorry I don't reply to DMs directly you need to use slash commands. e.g."
    )
    await send(ctx, bot_help_text)


@app.command()
def run_bot():
    bot.run(get_bot_token("IGBLOG-DISCORD-BOT"))


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    global bot_help_text
    bot_help_text = """```

Commands:
 /ask - Ask a blog questions
 - More coming ...
    ```"""


@bot.command(description="Show help")
async def help(ctx):
    response = f"{bot_help_text}"
    await ctx.respond(response)


@bot.command(name="ask", description="Message the bot")
async def ask_discord_command(ctx, question: str):
    await ctx.defer()
    await send(ctx, f"User asked: {question}")
    progress_bar_task = await draw_progress_bar(ctx)
    response = await iask(question, facts=10, u4=True, debug=False)
    ic(response)
    progress_bar_task.cancel()
    ic(response)
    await ctx.respond(response)


@bot.command(name="debug", description="Get Debgu Info from last call")
async def debug(ctx):
    await ctx.defer()
    await ctx.respond("Todo implement this")


# @logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()