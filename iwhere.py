#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "icecream",
#     "rich",
#     "langchain",
#     "langchain-core",
#     "langchain-openai",
#     "langchain-chroma",
#     "openai",
#     "pydantic",
#     "fastapi",
#     "requests",
#     "typing-extensions",
#     "chromadb",
#     "tiktoken",
#     "loguru",
#     "tqdm",
# ]
# ///

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
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
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
from langchain_core.language_models.chat_models import BaseChatModel
import json
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import sys
import time


# Directories to exclude from indexing
EXCLUDED_DIRS = [
    "zz-chop-logs",
    "chop-logs",
    "cursor-logs",
    "__pycache__",
    ".git",
    ".venv",
    ".cursor",
]

class LocationRecommendation(BaseModel):
    location: str
    markdown_path: str  # The markdown file path to edit
    reasoning: str


class BlogPlacementSuggestion(BaseModel):
    primary_location: LocationRecommendation
    alternative_locations: List[LocationRecommendation]
    structuring_tips: List[str]
    organization_tips: List[str]


class DebugInfo(BaseModel):
    documents: List[Document] = []
    question: str = ""
    count_tokens: int = 0
    model: str = ""


class TimingStats:
    """Class to track timing information for various stages of processing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.rag_start = 0
        self.rag_end = 0
        self.llm_start = 0
        self.llm_end = 0
        self.end_time = 0
        self.model_name = ""
        self.token_count = 0
        
    def start_rag(self):
        self.rag_start = time.time()
        
    def end_rag(self):
        self.rag_end = time.time()
        
    def start_llm(self):
        self.llm_start = time.time()
        
    def end_llm(self):
        self.llm_end = time.time()
        
    def finish(self):
        self.end_time = time.time()
        
    def print_stats(self):
        """Print timing statistics to stderr."""
        rag_time = self.rag_end - self.rag_start if self.rag_end > 0 else 0
        llm_time = self.llm_end - self.llm_start if self.llm_end > 0 else 0
        total_time = self.end_time - self.start_time
        
        stats = f"""
=== Performance Statistics ===
Model: {self.model_name}
RAG retrieval time: {rag_time:.2f}s
LLM inference time: {llm_time:.2f}s
Total processing time: {total_time:.2f}s
Token count: {self.token_count}
============================
"""
        print(stats, file=sys.stderr)


gpt_model = setup_gpt()
server = FastAPI()

app = typer.Typer(no_args_is_help=True)
console = Console()

CHROMA_DB_NAME = "blog.chroma.db"
DEFAULT_CHROMA_DB_DIR = CHROMA_DB_NAME
ALTERNATE_CHROMA_DB_DIR = os.path.expanduser(f"~/gits/nlp/{CHROMA_DB_NAME}")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
g_tracer: Optional[LangChainTracer] = None
g_debug_info = DebugInfo()

# Global variables to store database and documents
g_blog_content_db = None
g_all_documents = None

# Adjust to a safer size that won't exceed token limits
# OpenAI has a limit of 300,000 tokens, so we'll use a much smaller chunk size
chunk_size_tokens = 2000  # Much smaller to prevent hitting API limits


def get_chroma_db():
    global g_blog_content_db, g_all_documents
    
    # Return cached DB if it exists
    if g_blog_content_db is not None:
        return g_blog_content_db

    if os.path.exists(DEFAULT_CHROMA_DB_DIR):
        db_dir = DEFAULT_CHROMA_DB_DIR
    else:
        ic(f"Using alternate blog database location: {ALTERNATE_CHROMA_DB_DIR}")
        db_dir = ALTERNATE_CHROMA_DB_DIR

    if not os.path.exists(db_dir):
        if "build" in sys.argv:
            # If we're running the build command, return None
            return None
        raise Exception(
            f"Blog database not found in {DEFAULT_CHROMA_DB_DIR} or {ALTERNATE_CHROMA_DB_DIR}. Please run 'iwhere.py build' first."
        )

    g_blog_content_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    g_all_documents = g_blog_content_db.get()
    return g_blog_content_db


def should_exclude_path(path_str):
    """Check if the path should be excluded from indexing"""
    path = Path(path_str)
    
    # Check if any part of the path matches an excluded directory
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            logger.debug(f"Excluding path: {path_str} (contains excluded directory: {part})")
            return True
    return False


def get_blog_content(path="~/blog"):
    # set_trace()
    repo_path = pathlib.Path(os.path.expanduser(path))
    logger.info(f"Scanning for markdown files in {repo_path}")
    logger.info(f"Excluding directories: {EXCLUDED_DIRS}")
    
    # Find all markdown files, excluding specified directories
    markdown_files = []
    skipped_dirs = set()
    
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        original_dirs = dirs.copy()
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        # Track which directories were skipped
        skipped = set(original_dirs) - set(dirs)
        if skipped:
            skipped_dirs.update(skipped)
            logger.debug(f"Skipping directories at {root}: {', '.join(skipped)}")
        
        # Add markdown files
        for file in files:
            if file.endswith('.md'):
                full_path = Path(root) / file
                if not should_exclude_path(str(full_path)):
                    markdown_files.append(full_path)
    
    logger.info(f"Found {len(markdown_files)} markdown files for indexing")
    if skipped_dirs:
        logger.info(f"Skipped directories: {', '.join(skipped_dirs)}")
    
    # Process each markdown file
    for markdown_file in markdown_files:
        try:
            with open(markdown_file, "r", encoding="utf-8") as f:
                yield Document(
                    page_content=f.read(),
                    metadata={"source": str(markdown_file.relative_to(repo_path))},
                )
        except Exception as e:
            logger.error(f"Error reading {markdown_file}: {e}")


def chunk_documents_recursive(documents, chunk_size=chunk_size_tokens):
    """Split documents using recursive character splitting with token limit checks"""
    recursive_splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_size // 4
    )
    splitter = recursive_splitter
    
    total_skipped = 0

    for document in documents:
        try:
            # Skip very large documents that would cause issues
            doc_token_count = num_tokens_from_string(document.page_content)
            if doc_token_count > 100000:  # If document is extremely large (100k tokens)
                logger.warning(f"Document too large ({doc_token_count} tokens): {document.metadata.get('source', 'unknown')}. Using smaller chunks.")
                # Try to split with a very small chunk size for large documents
                chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                ).split_text(document.page_content)
            else:
                chunks = splitter.split_text(document.page_content)
                
            for chunk in chunks:
                # Estimate tokens and skip if too large
                token_count = num_tokens_from_string(chunk)
                if token_count > 8000:  # OpenAI's embeddings limit per chunk
                    logger.warning(f"Chunk too large: {token_count} tokens. Skipping.")
                    total_skipped += 1
                    continue
                    
                d = Document(
                    page_content=chunk,
                    metadata={
                        "chunk_method": "recursive_char",
                        "source": document.metadata["source"],
                        "is_entire_document": len(chunks) == 1,
                    },
                )
                yield d
        except Exception as e:
            logger.error(f"Error chunking document {document.metadata.get('source', 'unknown')}: {e}")
    
    if total_skipped > 0:
        logger.warning(f"Skipped {total_skipped} chunks that exceeded token limits")


def chunk_documents_as_md(documents, chunk_size=chunk_size_tokens):
    """Split documents using Markdown headers with token limit checks"""
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
    
    total_skipped = 0

    for document in documents:
        try:
            # Skip if not valid markdown or too short
            if not document.page_content or len(document.page_content) < 10:
                logger.warning(f"Document too short or empty: {document.metadata.get('source', 'unknown')}. Skipping.")
                continue
                
            # Check if document content contains markdown headers
            has_headers = any(header[0] in document.page_content for header in headers_to_split_on)
            
            # If no headers, use recursive splitting instead
            if not has_headers:
                logger.debug(f"No markdown headers found in {document.metadata.get('source', 'unknown')}. Using recursive splitting.")
                chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_size // 4
                ).split_text(document.page_content)
                
                for chunk in chunks:
                    token_count = num_tokens_from_string(chunk)
                    if token_count > 8000:
                        total_skipped += 1
                        continue
                        
                    yield Document(
                        page_content=chunk,
                        metadata={
                            "source": document.metadata["source"],
                            "chunk_method": "recursive_fallback",
                            "is_entire_document": len(chunks) == 1,
                        },
                    )
                continue
            
            try:
                md_chunks = splitter.split_text(document.page_content)
            except Exception as e:
                logger.warning(f"Error splitting markdown in {document.metadata.get('source', 'unknown')}: {e}. Falling back to recursive splitting.")
                # Fall back to recursive splitting
                chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_size // 4
                ).split_text(document.page_content)
                
                for chunk in chunks:
                    token_count = num_tokens_from_string(chunk)
                    if token_count > 8000:
                        total_skipped += 1
                        continue
                        
                    yield Document(
                        page_content=chunk,
                        metadata={
                            "source": document.metadata["source"],
                            "chunk_method": "recursive_fallback",
                            "is_entire_document": len(chunks) == 1,
                        },
                    )
                continue
                
            base_metadata = {
                "source": document.metadata["source"],
                "chunk_method": "md_simple",
                "is_entire_document": False,
            }
            
            for chunk in md_chunks:
                # Estimate tokens and skip if too large
                token_count = num_tokens_from_string(chunk.page_content)
                if token_count > 8000:  # OpenAI's embeddings limit per chunk
                    logger.warning(f"MD chunk too large: {token_count} tokens. Skipping.")
                    total_skipped += 1
                    continue
                    
                yield Document(
                    page_content=chunk.page_content,
                    metadata={**chunk.metadata, **base_metadata},
                )
        except Exception as e:
            logger.error(f"Error chunking MD document {document.metadata.get('source', 'unknown')}: {e}")
    
    if total_skipped > 0:
        logger.warning(f"Skipped {total_skipped} chunks that exceeded token limits")


def chunk_documents_as_md_large(documents, chunk_size=chunk_size_tokens):
    """Create larger chunks from markdown documents by merging smaller chunks up to token limits"""
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
    
    total_skipped = 0
    total_merged = 0

    for document in documents:
        try:
            # Skip if not valid markdown or too short
            if not document.page_content or len(document.page_content) < 10:
                logger.warning(f"Document too short or empty: {document.metadata.get('source', 'unknown')}. Skipping.")
                continue
                
            # Check if document is too large to process efficiently
            doc_token_count = num_tokens_from_string(document.page_content)
            if doc_token_count > 100000:  # If document is extremely large
                logger.warning(f"Document too large for MD large chunking ({doc_token_count} tokens): {document.metadata.get('source', 'unknown')}. Skipping.")
                continue
            
            # Check if document content contains markdown headers
            has_headers = any(header[0] in document.page_content for header in headers_to_split_on)
            
            # If no headers, skip for this chunker (handled by other chunkers)
            if not has_headers:
                logger.debug(f"No markdown headers found in {document.metadata.get('source', 'unknown')}. Skipping for MD large chunker.")
                continue
                
            # Try to split with markdown headers
            try:
                md_chunks = list(splitter.split_text(document.page_content))
            except Exception as e:
                logger.warning(f"Error splitting markdown in {document.metadata.get('source', 'unknown')}: {e}. Skipping.")
                continue
                
            # If we couldn't get any chunks, skip
            if not md_chunks:
                logger.warning(f"No chunks created for {document.metadata.get('source', 'unknown')}. Skipping.")
                continue
                
            base_metadata = {
                "source": document.metadata["source"],
                "chunk_method": "md_merge",
            }
            
            # Maximum token limit for OpenAI embeddings
            MAX_TOKENS = 8000
            # Target size for chunks (80% of max to leave room for headers)
            TARGET_SIZE = int(MAX_TOKENS * 0.8)
            
            candidate_chunk = Document(page_content="", metadata=base_metadata.copy())
            is_entire_document = True
            
            # Group chunks to approach but not exceed token limits
            for chunk in md_chunks:
                chunk_token_count = num_tokens_from_string(chunk.page_content)
                
                # If the current chunk alone is too big, skip it
                if chunk_token_count > MAX_TOKENS:
                    logger.warning(f"Single MD chunk too large: {chunk_token_count} tokens. Skipping.")
                    total_skipped += 1
                    continue
                    
                # Check if adding this chunk would exceed our limit
                current_tokens = num_tokens_from_string(candidate_chunk.page_content)
                combined_tokens = current_tokens + chunk_token_count
                
                # If adding would exceed limit, yield current chunk and start new one
                if current_tokens > 0 and combined_tokens > TARGET_SIZE:
                    is_entire_document = False
                    candidate_chunk.metadata["is_entire_document"] = is_entire_document
                    total_merged += 1
                    yield candidate_chunk
                    # Start a new chunk
                    candidate_chunk = Document(page_content="", metadata=base_metadata.copy())
                
                # Add content to current chunk
                if candidate_chunk.page_content:
                    candidate_chunk.page_content += "\n\n"
                candidate_chunk.page_content += chunk.page_content
                
                # If we have headers in metadata, merge them
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if key not in candidate_chunk.metadata:
                            candidate_chunk.metadata[key] = value
            
            # Don't forget to yield the last chunk
            if candidate_chunk.page_content:
                candidate_chunk.metadata["is_entire_document"] = is_entire_document
                total_merged += 1
                yield candidate_chunk
                
        except Exception as e:
            logger.error(f"Error in MD large chunking for {document.metadata.get('source', 'unknown')}: {e}")
    
    logger.info(f"MD large chunking: created {total_merged} merged chunks, skipped {total_skipped} oversized chunks")


def dedup_chunks(chunks):
    """Remove duplicate chunks to avoid redundant embeddings and storage
    
    If we have multiple chunks from the same source and some contain the full document,
    we'll prioritize keeping the full document and removing duplicates.
    """
    unique_chunks = []
    seen_full_size = set()
    content_hashes = set()  # To detect content duplicates
    total = len(chunks)
    missing_metadata = 0
    
    for chunk in chunks:
        # Check for missing required metadata
        if "source" not in chunk.metadata or "is_entire_document" not in chunk.metadata:
            logger.warning(f"Chunk has incomplete metadata: {chunk.metadata}")
            missing_metadata += 1
            continue
        
        source = chunk.metadata["source"]
        whole_doc = chunk.metadata["is_entire_document"]
        
        # Skip if we already have a full version of this document
        if whole_doc and source in seen_full_size:
            continue
            
        # Track entire documents we've seen
        if whole_doc:
            seen_full_size.add(source)
            
        # Create a hash of content to avoid exact duplicates with different metadata
        content = chunk.page_content.strip()
        if not content:  # Skip empty content
            continue
            
        # Use a simple hash for deduplication - trim to avoid whitespace differences
        content_hash = hash(content[:1000] + content[-1000:] if len(content) > 2000 else content)
        
        if content_hash in content_hashes:
            continue
            
        content_hashes.add(content_hash)
        unique_chunks.append(chunk)
    
    deduped = len(unique_chunks)
    logger.info(f"Deduplication: {total} → {deduped} chunks ({total - deduped} duplicates removed)")
    if missing_metadata > 0:
        logger.warning(f"Skipped {missing_metadata} chunks with missing metadata")
        
    return unique_chunks


def process_chunks_in_batches(chunks, batch_size=50):
    """Process chunks in smaller batches to avoid hitting API limits"""
    from tqdm import tqdm
    
    all_chunks = list(chunks)
    total_chunks = len(all_chunks)
    logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Processing batches"):
        batch = all_chunks[i:i+batch_size]
        yield batch
        

@app.command()
def build(
    blog_path: Annotated[str, typer.Option(help="Path to the blog repository")] = "~/blog",
    batch_size: Annotated[int, typer.Option(help="Number of chunks to process in each batch")] = 50,
):
    """Build the vector database from blog content, processing in batches to avoid API limits"""
    logger.info(f"Building blog database from {blog_path}")
    docs = list(get_blog_content(blog_path))
    logger.info(f"Loaded {len(docs)} documents")
    
    # Create all chunks first
    logger.info("Chunking documents with different methods...")
    chunks = []
    chunks.extend(list(chunk_documents_as_md(docs)))
    logger.info(f"Created {len(chunks)} markdown chunks")
    
    chunks.extend(list(chunk_documents_as_md_large(docs)))
    logger.info(f"Added large markdown chunks, total now: {len(chunks)}")
    
    chunks.extend(list(chunk_documents_recursive(docs)))
    logger.info(f"Added recursive chunks, total now: {len(chunks)}")
    
    deduped_chunks = dedup_chunks(chunks)
    logger.info(f"After deduplication: {len(deduped_chunks)} chunks (removed {len(chunks) - len(deduped_chunks)} duplicates)")
    
    # Create database directory if it doesn't exist
    os.makedirs(DEFAULT_CHROMA_DB_DIR, exist_ok=True)
    
    # Process in batches to avoid hitting API limits
    db = None
    logger.info(f"Processing chunks in batches of {batch_size}")
    try:
        for i, batch in enumerate(process_chunks_in_batches(deduped_chunks, batch_size=batch_size)):
            logger.info(f"Processing batch {i+1} with {len(batch)} chunks")
            if i == 0:
                # Create new db with first batch
                logger.info("Creating new Chroma database with first batch")
                db = Chroma.from_documents(
                    batch, embeddings, persist_directory=DEFAULT_CHROMA_DB_DIR
                )
            else:
                # Add to existing db for subsequent batches
                if db is None:
                    logger.info("Reconnecting to existing Chroma database")
                    db = Chroma(persist_directory=DEFAULT_CHROMA_DB_DIR, embedding_function=embeddings)
                
                logger.info(f"Adding batch {i+1} to database")
                db.add_documents(batch)
                
            logger.info(f"Completed batch {i+1}")
            
            # No need to call db.persist() - newer versions of Chroma auto-persist
            
    except Exception as e:
        logger.error(f"Error during database building: {e}")
        raise
    
    logger.info("Successfully built the blog database.")
    print("✅ Blog database successfully built and saved to", DEFAULT_CHROMA_DB_DIR)


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
    db = get_chroma_db()
    if db is None:
        return False
    all_documents = db.get()
    for m in all_documents["metadatas"]:
        if m["source"] == path and m["is_entire_document"]:
            return True
    return False


def get_document(path) -> Document:
    db = get_chroma_db()
    if db is None:
        raise Exception(f"Blog database not found. Please run 'iwhere.py build' first.")
    all_documents = db.get()
    for i, m in enumerate(all_documents["metadatas"]):
        if m["source"] == path and m["is_entire_document"]:
            return Document(page_content=all_documents["documents"][i], metadata=m)
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
    facts: Annotated[int, typer.Option(help="Number of documents to use for context")] = 20,
    debug: bool = typer.Option(True),
    model: Annotated[
        str, 
        typer.Option(
            help="Model to use: openai, claude, llama, google, etc. See langchain_helper.py for all options"
        )
    ] = "openai",
):
    response = asyncio.run(iask(question, facts, debug, model))
    print(response)


@app.command()
def where(
    topic: Annotated[str, typer.Argument(help="Topic to find placement for")],
    num_docs: Annotated[int, typer.Option(help="Number of documents to use for context")] = 20,
    debug: Annotated[bool, typer.Option(help="Show debugging information")] = False,
    model: Annotated[
        str, 
        typer.Option(
            help="Model to use: openai, claude, llama, google, etc. See langchain_helper.py for all options"
        )
    ] = "openai",
):
    """Suggest where to add new blog content about a topic"""
    response = asyncio.run(iask_where(topic, num_docs, debug, model))
    print(response)


async def iask_where(topic: str, num_docs: int = 20, debug: bool = False, model: str = "openai"):
    # Initialize timing stats
    timing = TimingStats()
    
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert blog organization consultant. You help Igor organize his blog content effectively.
Use chain of thought reasoning to suggest where new content about a topic should be added.

Topic to add: {topic}

Here is the current layout of the blog
<blog_information>
    {backlinks}
</blog_information>

Here is the current blog structure and content for reference:

<blog_chunks>
{context}
</blog_chunks>

Think through this step by step:
1. What is the main theme/purpose of this content?
2. What existing categories/sections might be relevant?
3. Are there similar topics already covered somewhere?
4. Should this be its own post or part of existing content?

Return your response as a JSON object matching this Pydantic model:

```python
class LocationRecommendation:
    location: str      # Where to put the content (section name/header)
    markdown_path: str # The full markdown file path (e.g. "_d/joy.md" or "_posts/something.md")
    reasoning: str     # Why this location makes sense

class BlogPlacementSuggestion:
    primary_location: LocationRecommendation
    alternative_locations: List[LocationRecommendation]
    structuring_tips: List[str]    # List of tips for content structure
    organization_tips: List[str]    # List of tips for organization
```

Ensure your response is valid JSON that matches this schema exactly.
When suggesting locations, always include both the section within the file and the complete markdown file path relative to the blog root.
File paths should always start with either "_d/" or "_posts/".
    """
    )

    llm = get_model_for_name(model)
    model_name = langchain_helper.get_model_name(llm)
    timing.model_name = model_name
    
    if debug:
        ic(f"Using model: {model_name}")
    
    # Time the RAG process
    timing.start_rag()
    
    db = get_chroma_db()
    if db is None:
        raise Exception(f"Blog database not found. Please run 'iwhere.py build' first.")
    
    logger.info(f"Searching for documents related to '{topic}' using {num_docs} documents")
    docs_and_scores = await db.asimilarity_search_with_relevance_scores(
        topic, k=num_docs
    )
    
    timing.end_rag()
    
    if debug:
        rag_time = timing.rag_end - timing.rag_start
        ic(f"RAG retrieval completed in {rag_time:.2f} seconds")
        ic("Retrieved documents and scores:")
        for doc, score in docs_and_scores:
            ic(doc.metadata, score)

    facts_to_inject = [doc for doc, _ in docs_and_scores]
    context = docs_to_prompt(facts_to_inject)
    timing.token_count = num_tokens_from_string(context)

    from langchain.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=BlogPlacementSuggestion)

    chain = prompt | llm | parser
    backlinks_content = (
        Path.home() / "gits/idvorkin.github.io/back-links.json"
    ).read_text()
    
    # Time the LLM inference - only the actual API call
    timing.start_llm()
    result = await chain.ainvoke(
        {"topic": topic, "context": context, "backlinks": backlinks_content}
    )
    timing.end_llm()
    
    if debug:
        llm_time = timing.llm_end - timing.llm_start
        ic(f"LLM inference completed in {llm_time:.2f} seconds")
        ic("LLM Response:", result)

    response = f"""
RECOMMENDED LOCATIONS:

PRIMARY LOCATION:
File Path: {result.primary_location.markdown_path}
Location: {result.primary_location.location}
Reasoning: {result.primary_location.reasoning}

ALTERNATIVE LOCATIONS:

{chr(10).join(f'''Location {i+1}:
File Path: {loc.markdown_path}
Location: {loc.location}
Reasoning: {loc.reasoning}
''' for i, loc in enumerate(result.alternative_locations))}

ADDITIONAL SUGGESTIONS:

Structuring Tips:
{chr(10).join(f'• {tip}' for tip in result.structuring_tips)}

Organization Tips:
{chr(10).join(f'• {tip}' for tip in result.organization_tips)}
"""
    
    # Finish timing and print stats
    timing.finish()
    timing.print_stats()
    
    return response


async def iask(
    question: str,
    facts: int = 20,
    debug: bool = True,
    model: str = "openai",
):
    # Initialize timing stats
    timing = TimingStats()
    
    if debug:
        ic(facts)
    
    # load chroma from DB
    db = get_chroma_db()
    if db is None:
        raise Exception(f"Blog database not found. Please run 'iwhere.py build' first.")

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

    llm = get_model_for_name(model)
    model_name = langchain_helper.get_model_name(llm)
    timing.model_name = model_name
    
    if debug:
        ic(f"Using model: {model_name}")

    # Time RAG retrieval
    timing.start_rag()
    
    docs_and_scores = await db.asimilarity_search_with_relevance_scores(
        question, k=4 * facts
    )
    
    timing.end_rag()
    
    if debug:
        rag_time = timing.rag_end - timing.rag_start
        ic(f"RAG retrieval completed in {rag_time:.2f} seconds")
        for doc, score in docs_and_scores:
            ic(doc.metadata, score)

    candidate_facts = [d for d, _ in docs_and_scores]

    facts_to_inject: List[Document] = []
    
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
            if debug:
                ic("Whole file already present", fact_path)
            continue
        # Whole document is available
        if has_whole_document(fact_path):
            if debug:
                ic("Adding whole file instead", fact.metadata)
            facts_to_inject.append(get_document(fact_path))
            continue
        # All we have is the partial
        facts_to_inject.append(fact)

    good_docs = ["_posts/2020-04-01-Igor-Eulogy.md", "_d/operating-manual-2.md"]
    facts_to_inject += [get_document(d) for d in good_docs]

    if debug:
        print("Source Documents")
        for doc in facts_to_inject:
            # Remap metadata to url
            ic(doc.metadata)

    context = docs_to_prompt(facts_to_inject)
    timing.token_count = num_tokens_from_string(context)
    
    if debug:
        ic(timing.token_count)
    
    chain = prompt | llm | StrOutputParser()
    global g_debug_info
    g_debug_info = DebugInfo()
    g_debug_info.documents = facts_to_inject
    g_debug_info.count_tokens = timing.token_count
    g_debug_info.question = question
    g_debug_info.model = model_name

    # Time the actual LLM inference
    timing.start_llm()
    response = await chain.ainvoke({"question": question, "context": context})
    timing.end_llm()
    
    if debug:
        llm_time = timing.llm_end - timing.llm_start
        ic(f"LLM inference completed in {llm_time:.2f} seconds")

    # Finish timing and print stats
    timing.finish()
    timing.print_stats()
    
    return response


def get_model_for_name(model_name: str) -> BaseChatModel:
    """Convert model name to the appropriate model object"""
    model_name = model_name.lower()
    
    if model_name == "openai":
        return langchain_helper.get_model(openai=True)
    elif model_name == "claude":
        return langchain_helper.get_model(claude=True)
    elif model_name == "llama":
        return langchain_helper.get_model(llama=True)
    elif model_name == "google":
        return langchain_helper.get_model(google=True)
    elif model_name == "google_think":
        return langchain_helper.get_model(google_think=True)
    elif model_name == "google_flash":
        return langchain_helper.get_model(google_flash=True)
    elif model_name == "deepseek":
        return langchain_helper.get_model(deepseek=True)
    elif model_name == "o4_mini":
        return langchain_helper.get_model(o4_mini=True)
    elif model_name == "openai_mini":
        return langchain_helper.get_model(openai_mini=True)
    else:
        logger.warning(f"Unknown model name: {model_name}, defaulting to OpenAI")
        return langchain_helper.get_model(openai=True)


def app_wrap_loguru():
    """Configure logging with loguru for console and file output"""
    logger.remove()  # Remove default handler
    # Set up console logging with color
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", 
        level="DEBUG",
        colorize=True
    )
    logger.info("Starting application")
    return app()


if __name__ == "__main__":
    app_wrap_loguru()
