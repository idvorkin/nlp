#!python3
import asyncio
from langchain.callbacks.tracers.langchain import LangChainTracer
import requests
from functools import lru_cache
import pathlib
from rich.console import Console
from icecream import ic
import typer
import os
from rich import print
from typing import List, Optional
import time
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import langchain_helper
from langchain import (
    text_splitter,
)  # import CharacterTextSplitter, RecursiveCharacterTextSplitter, Markdown
from typing_extensions import Annotated
from fastapi import FastAPI
from openai_wrapper import setup_gpt, num_tokens_from_string
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
import json
from pydantic import BaseModel
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class LocationRecommendation(BaseModel):
    location: str
    markdown_path: str  # The markdown file path to edit
    reasoning: str


class BlogPlacementSuggestion(BaseModel):
    primary_location: LocationRecommendation
    alternative_locations: List[LocationRecommendation]
    structuring_tips: List[str]
    organization_tips: List[str]


gpt_model = setup_gpt()
server = FastAPI()

app = typer.Typer(no_args_is_help=True)
console = Console()


# Configuration constants
class Config:
    # Index storage
    FAISS_DB_DIR = "blog.faiss"
    BM25_FILENAME = "bm25.pkl"

    # Processing parameters
    BATCH_SIZE = 100  # Documents per batch for embedding
    CHUNK_SIZE_TOKENS = 4 * 1000 * 5  # ~5K tokens
    MAX_FILE_SIZE = 500000  # Max chars per file (prevent huge files)

    # Search parameters
    FACT_MULTIPLIER = 4  # Multiply facts by this for initial retrieval
    BM25_WEIGHT = 0.3
    FAISS_WEIGHT = 0.7

    # Allowed directories
    ALLOWED_DIRS = ["_d", "_posts", "_ig66", "_td"]


# Legacy compatibility
FAISS_DB_DIR = Config.FAISS_DB_DIR
DEFAULT_FAISS_DB_DIR = Config.FAISS_DB_DIR
ALTERNATE_FAISS_DB_DIR = os.path.expanduser(f"~/gits/nlp/{Config.FAISS_DB_DIR}")
BM25_FILENAME = Config.BM25_FILENAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
g_tracer: Optional[LangChainTracer] = None
# embeddings = OpenAIEmbeddings()


def get_faiss_db():
    if os.path.exists(DEFAULT_FAISS_DB_DIR):
        db_dir = DEFAULT_FAISS_DB_DIR
    else:
        ic(f"Using alternate blog database location: {ALTERNATE_FAISS_DB_DIR}")
        db_dir = ALTERNATE_FAISS_DB_DIR

    if not os.path.exists(db_dir):
        raise Exception(
            f"Blog database not found in {DEFAULT_FAISS_DB_DIR} or {ALTERNATE_FAISS_DB_DIR}"
        )

    return FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)


def get_bm25_retriever():
    """Load BM25 retriever from pickle file in FAISS directory"""
    # Check primary location
    primary_bm25 = os.path.join(DEFAULT_FAISS_DB_DIR, BM25_FILENAME)
    alternate_bm25 = os.path.join(ALTERNATE_FAISS_DB_DIR, BM25_FILENAME)

    if os.path.exists(primary_bm25):
        bm25_file = primary_bm25
    elif os.path.exists(alternate_bm25):
        ic(f"Using alternate BM25 location: {alternate_bm25}")
        bm25_file = alternate_bm25
    else:
        raise Exception(
            f"BM25 index not found in {primary_bm25} or {alternate_bm25}. Run 'build' command first."
        )

    with open(bm25_file, "rb") as f:
        return pickle.load(f)


def save_bm25_retriever(bm25_retriever):
    """Save BM25 retriever to pickle file in FAISS directory"""
    bm25_path = os.path.join(DEFAULT_FAISS_DB_DIR, BM25_FILENAME)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    ic(f"BM25 index saved to {bm25_path}")


def get_hybrid_retriever(k: int = 10):
    """Get ensemble retriever combining FAISS and BM25"""
    faiss_db = get_faiss_db()
    bm25_retriever = get_bm25_retriever()

    # Create retrievers with specified k
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": k})

    # Ensemble with configurable weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[Config.BM25_WEIGHT, Config.FAISS_WEIGHT],
    )

    return ensemble_retriever


class DebugInfo(BaseModel):
    documents: List[Document] = []
    question: str = ""
    count_tokens: int = 0
    model: str = ""


g_debug_info = DebugInfo()


def chunk_documents_recursive(documents, chunk_size=Config.CHUNK_SIZE_TOKENS):
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


def chunk_documents_as_md(documents, chunk_size=Config.CHUNK_SIZE_TOKENS):
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


def chunk_documents_as_md_large(documents, chunk_size=Config.CHUNK_SIZE_TOKENS):
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

    # Only include whitelisted directories
    markdown_files = []

    for dir_name in Config.ALLOWED_DIRS:
        dir_path = repo_path / dir_name
        if dir_path.exists():
            markdown_files.extend(dir_path.glob("*.md"))

    for markdown_file in markdown_files:
        with open(markdown_file, "r") as f:
            content = f.read()
            # Skip if file is somehow massive (shouldn't happen with normal blog posts)
            if len(content) > Config.MAX_FILE_SIZE:
                ic(f"Skipping massive file {markdown_file}: {len(content)} chars")
                continue
            yield Document(
                page_content=content,
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
    # db_path = pathlib.Path(FAISS_DB_DIR)
    # db_path.rmdir()

    ic(len(docs))
    chunks = list(chunk_documents_as_md(docs))
    chunks += list(chunk_documents_as_md_large(docs))
    chunks += list(chunk_documents_recursive(docs))
    deduped_chunks = dedup_chunks(chunks)
    ic(len(chunks), len(deduped_chunks))

    # Build FAISS index in batches to avoid token limits
    print("Building FAISS index in batches...")
    db = None

    for i in range(0, len(deduped_chunks), Config.BATCH_SIZE):
        batch = deduped_chunks[i : i + Config.BATCH_SIZE]
        batch_num = i // Config.BATCH_SIZE + 1
        total_batches = (
            len(deduped_chunks) + Config.BATCH_SIZE - 1
        ) // Config.BATCH_SIZE
        print(f"Processing batch {batch_num}/{total_batches}")

        if db is None:
            # Create initial FAISS index with first batch
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Add to existing index
            db.add_documents(batch)

    db.save_local(DEFAULT_FAISS_DB_DIR)
    ic(f"FAISS index saved to {DEFAULT_FAISS_DB_DIR}")

    # Build BM25 index and persist it
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(deduped_chunks)
    bm25_retriever.k = 10  # Set default k for BM25
    save_bm25_retriever(bm25_retriever)

    print(f"✅ Both indexes built successfully in {DEFAULT_FAISS_DB_DIR}/")
    print("  - FAISS: index.faiss, index.pkl")
    print(f"  - BM25: {BM25_FILENAME}")


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


# Lazy initialization to avoid race condition when DB doesn't exist
g_blog_content_db = None
g_all_documents = None


def get_or_init_blog_db():
    """Get or initialize the blog content database lazily"""
    global g_blog_content_db, g_all_documents
    if g_blog_content_db is None:
        try:
            g_blog_content_db = get_faiss_db()
            _docs = list(g_blog_content_db.docstore._dict.values())
            g_all_documents = {
                "documents": [d.page_content for d in _docs],
                "metadatas": [d.metadata for d in _docs],
            }
        except Exception as e:
            ic(f"Failed to initialize blog database: {e}")
            raise
    return g_blog_content_db, g_all_documents


async def get_blog_db_for_search(question: str, k: int, debug: bool = False):
    """
    DRY helper to get documents from blog database for similarity search.
    Used by both iask and iask_where functions.

    Args:
        question: The query string
        k: Number of documents to retrieve
        debug: Whether to show debug output

    Returns:
        Tuple of (docs_and_scores, blog_content_db)
    """
    blog_content_db, _ = get_or_init_blog_db()
    docs_and_scores = await blog_content_db.asimilarity_search_with_relevance_scores(
        question, k=k
    )
    if debug:
        ic("Retrieved documents and scores:")
        for doc, score in docs_and_scores:
            ic(doc.metadata, score)
    return docs_and_scores, blog_content_db


def has_whole_document(path):
    _, all_documents = get_or_init_blog_db()
    for m in all_documents["metadatas"]:
        if m.get("source") == path and m.get("is_entire_document"):
            return True
    return False


def get_document(path) -> Document:
    _, all_documents = get_or_init_blog_db()
    for i, m in enumerate(all_documents["metadatas"]):
        if m.get("source") == path and m.get("is_entire_document"):
            return Document(page_content=all_documents["documents"][i], metadata=m)
    raise Exception(f"{path} document not found")


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
    debug: bool = typer.Option(True),
    hybrid: bool = typer.Option(True, help="Use hybrid search (BM25 + FAISS)"),
    fast: bool = typer.Option(False, help="Use fast GPT-OSS model"),
):
    start_time = time.perf_counter()
    response, timings = asyncio.run(
        iask(question, facts, debug, use_hybrid=hybrid, use_fast=fast)
    )
    elapsed_time = time.perf_counter() - start_time

    print(response)
    timing_str = " | ".join(
        [f"{stage}: {duration:.2f}s" for stage, duration in timings.items()]
    )
    print(f"\n⏱️  {timing_str} | Total: {elapsed_time:.2f}s")


async def iask_where(topic: str, debug: bool = False, use_fast: bool = False):
    """Suggest where to add new blog content - simplified to match iask pattern"""
    timings = {}

    if debug:
        ic(f"Finding placement for topic: {topic}")
        if use_fast:
            ic("Using fast GPT-OSS model")

    # Simplified prompt without backlinks file dependency
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert blog organization consultant. You help Igor organize his blog content effectively.

Topic to add: {topic}

Based on the following blog content, suggest where this new content should be added:

<blog_chunks>
{context}
</blog_chunks>

Think through this step by step:
1. What is the main theme/purpose of this content?
2. What existing posts/sections are most relevant?
3. Should this be its own post or part of existing content?

Provide a clear recommendation with:
- PRIMARY LOCATION: The best place to add this content (file path and section)
- ALTERNATIVE LOCATIONS: 2-3 other good options
- REASONING: Why these locations make sense

Keep your response practical and actionable.
    """
    )

    # Model selection timing
    model_start = time.perf_counter()
    if use_fast:
        llm = langchain_helper.get_model(gpt_oss=True)
    else:
        llm = langchain_helper.get_model(openai=True)
    timings["Model initialization"] = time.perf_counter() - model_start

    # Document retrieval timing
    retrieval_start = time.perf_counter()
    # Use DRY helper for database access
    docs_and_scores, _ = await get_blog_db_for_search(topic, k=8, debug=debug)
    timings["Document retrieval"] = time.perf_counter() - retrieval_start

    # Context preparation timing
    context_start = time.perf_counter()
    facts_to_inject = [doc for doc, _ in docs_and_scores]
    context = docs_to_prompt(facts_to_inject)
    timings["Context preparation"] = time.perf_counter() - context_start

    if debug:
        ic(f"Context tokens: {num_tokens_from_string(context)}")

    chain = prompt | llm | StrOutputParser()

    # LLM inference timing
    llm_start = time.perf_counter()
    try:
        result = await chain.ainvoke({"topic": topic, "context": context})
        if debug:
            ic("LLM Response received")
    except Exception as e:
        if debug:
            ic(f"Error in iask_where: {e}")
        result = f"Error: Could not generate placement suggestion - {str(e)}"
    timings["LLM inference"] = time.perf_counter() - llm_start

    return result, timings


async def iask(
    question: str,
    facts: int,
    debug: bool = True,
    use_hybrid: bool = True,
    use_fast: bool = False,
):
    timings = {}

    if debug:
        ic(facts)
        ic(f"Using {'hybrid' if use_hybrid else 'FAISS-only'} retrieval")
        if use_fast:
            ic("Using fast GPT-OSS model")

    # load FAISS index from DB

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
* [Igor's Eulogy - Importance of smiling](/eulogy#smiling) - Igor's eulogy talks about how he always smiled (90%)
    """
    )

    # Model selection timing
    model_start = time.perf_counter()
    if use_fast:
        llm = langchain_helper.get_model(gpt_oss=True)
    else:
        llm = langchain_helper.get_model(openai=True)
    timings["Model initialization"] = time.perf_counter() - model_start

    # Document retrieval timing
    retrieval_start = time.perf_counter()

    # Use hybrid retrieval if enabled and BM25 index exists
    if use_hybrid:
        try:
            hybrid_retriever = get_hybrid_retriever(k=Config.FACT_MULTIPLIER * facts)
            docs = await hybrid_retriever.ainvoke(question)
            # Add dummy scores for compatibility with existing code
            docs_and_scores = [(doc, 0.0) for doc in docs]
            if debug:
                ic("Using hybrid retrieval (BM25 + FAISS)")
        except FileNotFoundError as e:
            if debug:
                ic(f"BM25 index not found, falling back to FAISS: {e}")
            use_hybrid = False
        except pickle.UnpicklingError as e:
            if debug:
                ic(f"BM25 index corrupted, falling back to FAISS: {e}")
            use_hybrid = False
        except Exception as e:
            if debug:
                ic(
                    f"Hybrid retrieval failed unexpectedly, falling back to FAISS: {type(e).__name__}: {e}"
                )
            use_hybrid = False

    if not use_hybrid:
        # Fall back to FAISS-only retrieval
        docs_and_scores, _ = await get_blog_db_for_search(
            question, k=Config.FACT_MULTIPLIER * facts, debug=False
        )
        if debug:
            ic("Using FAISS-only retrieval")

    timings["Document retrieval"] = time.perf_counter() - retrieval_start

    # Context preparation timing
    context_start = time.perf_counter()

    for doc, score in docs_and_scores:
        ic(doc.metadata, score)

    candidate_facts = [d for d, _ in docs_and_scores]

    facts_to_inject: List[Document] = []
    # build a set of facts to inject
    # if we got suggested partial files, try to find the full size version
    # if we can inject full size version, include that.
    # include upto fact docs

    def facts_to_append_contains_whole_file(path):
        for fact in facts_to_inject:
            if fact.metadata["source"] == path and fact.metadata["is_entire_document"]:
                return True
        return False

    # We can improve our relevance by getting the md_simple_chunks, but that loses context
    # Rebuild context by pulling in the largest chunk i can that contains the smaller chunk
    for fact in candidate_facts:
        if len(facts_to_inject) >= facts:
            break
        fact_path = fact.metadata["source"]
        # Already added
        if facts_to_append_contains_whole_file(fact_path):
            ic("Whole file already present", fact_path)
            continue
        # Whole document is available
        if has_whole_document(fact_path):
            ic("Adding whole file instead", fact.metadata)
            facts_to_inject.append(get_document(fact_path))
            continue
        # All we have is the partial
        facts_to_inject.append(fact)

    timings["Context preparation"] = time.perf_counter() - context_start

    # Skip adding good_docs for now since there's a path mismatch issue
    # good_docs = ["_posts/2020-04-01-Igor-Eulogy.md", "_d/operating-manual-2.md"]
    # facts_to_inject += [get_document(d) for d in good_docs]

    print("Source Documents")
    for doc in facts_to_inject:
        # Remap metadata to url
        ic(doc.metadata)

    context = docs_to_prompt(facts_to_inject)
    ic(num_tokens_from_string(context))
    chain = prompt | llm | StrOutputParser()
    global g_debug_info
    # dunno why this isn't working, to lazy to fix.
    # g_debug_info = DebugInfo(
    # documents = facts_to_inject,
    # count_tokens = num_tokens_from_string(context),
    # question = question
    # )
    g_debug_info = DebugInfo()
    g_debug_info.documents = facts_to_inject
    g_debug_info.count_tokens = num_tokens_from_string(context)
    g_debug_info.question = question
    g_debug_info.model = langchain_helper.get_model_name(llm)

    # LLM inference timing
    llm_start = time.perf_counter()
    response = await chain.ainvoke({"question": question, "context": context})
    timings["LLM inference"] = time.perf_counter() - llm_start

    return response, timings


# @logger.catch()
def app_wrap_loguru():
    app()


@app.command()
def where(
    topic: Annotated[str, typer.Argument(help="Topic to find placement for")],
    debug: Annotated[bool, typer.Option(help="Show debugging information")] = False,
    fast: Annotated[bool, typer.Option(help="Use fast GPT-OSS model")] = False,
):
    """Suggest where to add new blog content about a topic"""
    start_time = time.perf_counter()
    response, timings = asyncio.run(iask_where(topic, debug, use_fast=fast))
    elapsed_time = time.perf_counter() - start_time

    print(response)
    timing_str = " | ".join(
        [f"{stage}: {duration:.2f}s" for stage, duration in timings.items()]
    )
    print(f"\n⏱️  {timing_str} | Total: {elapsed_time:.2f}s")


if __name__ == "__main__":
    app_wrap_loguru()
