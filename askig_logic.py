import requests
from functools import lru_cache
import pathlib
from icecream import ic
import os
import json
from typing import List, Optional

from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser

from loguru import (
    logger,
)  # Assuming loguru is configured elsewhere or not needed for logic file directly
# If needed, logger configuration might be required.

import langchain_helper
from openai_wrapper import num_tokens_from_string
from askig_models import (
    TimingStats,
    DebugInfo,
    BlogPlacementSuggestion,
)  # Import models

# --- Constants and Globals ---
FAISS_DB_DIR = "blog.faiss"
DEFAULT_FAISS_DB_DIR = FAISS_DB_DIR  # Relative to where this script/module is run
ALTERNATE_FAISS_DB_DIR = os.path.expanduser(f"~/gits/nlp/{FAISS_DB_DIR}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
g_tracer: Optional[LangChainTracer] = (
    None  # Assuming LangChainTracer is set up elsewhere if used
)

g_blog_content_db = None
g_all_documents = None  # To store all docs from FAISS, avoiding repeated loads
g_debug_info = DebugInfo()  # Initialize with default DebugInfo


# --- Database Functions ---
def get_faiss_db():
    global g_blog_content_db, g_all_documents

    if g_blog_content_db is not None:
        if g_all_documents is None:  # Defensive check if DB is cached but docs aren't
            logger.warning(
                "g_blog_content_db is cached, but g_all_documents is None. Re-fetching documents."
            )
            g_all_documents = (
                g_blog_content_db.get()
            )  # This might be slow if called often
        return g_blog_content_db

    # Determine DB directory
    # This assumes iwhere.py (and its 'build' command) creates the DB in DEFAULT_FAISS_DB_DIR
    # or the user ensures it's in ALTERNATE_FAISS_DB_DIR.
    if os.path.exists(DEFAULT_FAISS_DB_DIR):
        db_dir = DEFAULT_FAISS_DB_DIR
    elif os.path.exists(ALTERNATE_FAISS_DB_DIR):
        logger.info(
            f"Using alternate blog database location: {ALTERNATE_FAISS_DB_DIR}"
        )
        db_dir = ALTERNATE_FAISS_DB_DIR
    else:
        # If this logic file is run standalone without iwhere.py's build having run,
        # or without the DB being in a known location, this will be an issue.
        # For an MCP server, it's assumed the DB exists.
        raise Exception(
            f"Blog database not found in {DEFAULT_FAISS_DB_DIR} or {ALTERNATE_FAISS_DB_DIR}. "
            "Please ensure the database is built and accessible."
        )

    logger.info(f"Loading FAISS database from: {db_dir}")
    g_blog_content_db = FAISS.load_local(db_dir, embeddings)
    # Fetch all documents once and store them
    logger.info("Fetching all documents from FAISS DB into memory...")
    docs = list(g_blog_content_db.docstore._dict.values())
    g_all_documents = {
        "documents": [d.page_content for d in docs],
        "metadatas": [d.metadata for d in docs],
    }
    logger.info(f"Fetched {len(g_all_documents['documents'])} documents into memory.")
    return g_blog_content_db


def has_whole_document(path: str) -> bool:
    global g_all_documents
    if g_all_documents is None:
        logger.warning(
            "g_all_documents is None in has_whole_document. Attempting to load DB."
        )
        get_faiss_db()  # Attempt to load
        if g_all_documents is None:  # Still None
            logger.error("Failed to load g_all_documents in has_whole_document.")
            return False

    for m in g_all_documents["metadatas"]:
        if m.get("source") == path and m.get("is_entire_document"):
            return True
    return False


def get_document(path: str) -> Document:
    global g_all_documents
    if g_all_documents is None:
        logger.warning(
            "g_all_documents is None in get_document. Attempting to load DB."
        )
        get_faiss_db()  # Attempt to load
        if g_all_documents is None:  # Still None
            logger.error(
                f"Failed to load g_all_documents. Cannot find document for path '{path}'."
            )
            raise Exception(
                f"Whole document for path '{path}' not found as g_all_documents is not populated."
            )

    for i, m in enumerate(g_all_documents["metadatas"]):
        if m.get("source") == path and m.get("is_entire_document"):
            return Document(page_content=g_all_documents["documents"][i], metadata=m)
    logger.error(
        f"Whole document for path '{path}' not found in pre-fetched g_all_documents."
    )
    raise Exception(f"Whole document for path '{path}' not found in g_all_documents.")


# --- Content Processing Functions ---
@lru_cache
def build_markdown_to_url_map():
    source_file_to_url = {}
    backlinks_url = "https://raw.githubusercontent.com/idvorkin/idvorkin.github.io/master/back-links.json"
    try:
        d = requests.get(backlinks_url).json()
        url_infos = d.get("url_info", {})
        source_file_to_url = {
            v["markdown_path"]: k for k, v in url_infos.items() if "markdown_path" in v
        }
    except Exception as e:
        logger.error(
            f"Error fetching or processing backlinks_url: {e}. Returning empty map."
        )
    return source_file_to_url


def fixup_markdown_path(src: str) -> str:
    markdown_to_url = build_markdown_to_url_map()
    # First, handle specific known patterns if any, then general markdown_to_url
    # Example: Fixup ig66 paths
    for i in range(100 * 52):  # Assuming this range is appropriate
        src = src.replace(
            f"_ig66/{i}.md", f"[Family Journal {i}](https://idvork.in/ig66/{i})"
        )

    # General replacement from backlinks
    for md_file_path, url in markdown_to_url.items():
        if (
            md_file_path in src
        ):  # Only replace if the exact md_file_path string is found
            # url might start with '/', ensure link is correct
            display_url = url[1:] if url.startswith("/") else url
            md_link = f"[{display_url}](https://idvork.in/{display_url})"
            src = src.replace(md_file_path, md_link)

    return src


def docs_to_prompt(docs: List[Document]) -> str:
    ic(len(docs))  # Keep ic for debugging if desired
    ret = []
    for d in docs:
        # Make a copy of metadata to avoid modifying the global g_all_documents cache
        metadata_copy = d.metadata.copy()
        metadata_copy["source"] = fixup_markdown_path(
            metadata_copy.get("source", "Unknown source")
        )
        ret.append({"content": d.page_content, "metadata": metadata_copy})
    return json.dumps(ret, indent=2)  # Adding indent for readability if debugged


# --- Model Functions ---
def get_model_for_name(model_name: str) -> BaseChatModel:
    model_name = model_name.lower()
    # Mapping to langchain_helper functions
    if model_name == "openai":
        return langchain_helper.get_model(openai=True)
    if model_name == "claude":
        return langchain_helper.get_model(claude=True)
    if model_name == "llama":
        return langchain_helper.get_model(llama=True)
    if model_name == "google":
        return langchain_helper.get_model(google=True)
    if model_name == "google_think":
        return langchain_helper.get_model(google_think=True)
    if model_name == "google_flash":
        return langchain_helper.get_model(google_flash=True)
    if model_name == "deepseek":
        return langchain_helper.get_model(deepseek=True)
    if model_name == "o4_mini":
        return langchain_helper.get_model(o4_mini=True)
    if model_name == "openai_mini":
        return langchain_helper.get_model(openai_mini=True)

    logger.warning(f"Unknown model name: {model_name}, defaulting to OpenAI.")
    return langchain_helper.get_model(openai=True)


# --- Core Logic ---
async def iask_logic(
    question: str, facts: int = 20, debug: bool = False, model: str = "openai"
) -> str:
    global g_debug_info  # Allow modification of the global
    timing = TimingStats()
    timing.start_overall_init()

    llm = get_model_for_name(model)
    model_name_used = langchain_helper.get_model_name(llm)
    timing.model_name = model_name_used
    if debug:
        ic(f"Using model: {model_name_used}", facts)
    timing.end_overall_init()

    timing.start_db_load()
    db = get_faiss_db()
    timing.end_db_load()
    if db is None:
        raise Exception(
            "Blog database (FAISS) not loaded."
        )  # Should be caught by get_faiss_db

    timing.start_rag()
    # Simulating k=4*facts as in original iwhere.py, then filtering.
    # Consider if the k value should be directly `facts` or if the multiplier is important.
    docs_and_scores = await db.asimilarity_search_with_relevance_scores(
        question, k=4 * facts
    )
    timing.end_rag()

    if debug:
        ic(
            f"RAG retrieval completed in {timing.rag_end - timing.rag_start:.2f} seconds"
        )
        for doc, score in docs_and_scores:
            ic(doc.metadata, score)

    timing.start_doc_prep()
    candidate_facts = [d for d, _ in docs_and_scores]
    facts_to_inject: List[Document] = []

    # Logic for selecting facts to inject (replicated from iwhere.py)
    added_sources_for_whole_doc = set()
    for fact_doc in candidate_facts:
        if len(facts_to_inject) >= facts:
            break
        fact_path = fact_doc.metadata.get("source")
        if not fact_path:
            continue

        is_whole_doc_available = has_whole_document(fact_path)

        if is_whole_doc_available and fact_path not in added_sources_for_whole_doc:
            if debug:
                ic("Adding whole file for source:", fact_path)
            facts_to_inject.append(get_document(fact_path))
            added_sources_for_whole_doc.add(fact_path)
        elif not is_whole_doc_available:
            # If whole doc is not available, or if it is but we already added it,
            # consider adding this specific chunk if its source isn't already covered by a whole doc.
            # This logic might need refinement to perfectly match original intent.
            # Current simple way: if not already covered by a whole doc by source, add chunk.
            is_source_covered_by_whole = any(
                f.metadata.get("source") == fact_path
                and f.metadata.get("is_entire_document")
                for f in facts_to_inject
            )
            if not is_source_covered_by_whole:
                if debug:
                    ic(
                        "Adding specific chunk for source:",
                        fact_path,
                        fact_doc.metadata,
                    )
                facts_to_inject.append(fact_doc)

    # Ensure 'good_docs' are added if specified (original logic)
    good_docs_paths = ["_posts/2020-04-01-Igor-Eulogy.md", "_d/operating-manual-2.md"]
    for gd_path in good_docs_paths:
        if gd_path not in added_sources_for_whole_doc and has_whole_document(gd_path):
            if len(facts_to_inject) < facts:  # Only add if we still have space
                if debug:
                    ic("Adding good doc (whole):", gd_path)
                facts_to_inject.append(get_document(gd_path))
                added_sources_for_whole_doc.add(gd_path)  # Mark as added
            else:
                if debug:
                    ic("Skipping good doc due to fact limit:", gd_path)

    if debug:
        logger.info("Final source documents for prompt:")
        for doc in facts_to_inject:
            ic(doc.metadata)

    context = docs_to_prompt(facts_to_inject)
    timing.token_count = num_tokens_from_string(context)
    timing.end_doc_prep()
    if debug:
        ic(f"Token count for context: {timing.token_count}")

    prompt_template = ChatPromptTemplate.from_template(
        """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
The content is all from Igor's blog.
If you don't know the answer, just say that you don't know. Keep the answer under 10 lines.

# The User's Questions
{question}

# Context
{context}

# Instruction
Your answer should include sources like those listed below. The source files are markdown so if they have a header make an HTML anchor link when you make the source link. E.g. if it's in idvork.in, with header # Foo , set it to http://idvork.in#foo

### Sources
* source file link - Your reasoning on why it's relevant (% relevance, e.g. 20%)
* [Igor's Eulogy - Importance of smiling](/eulogy#smiling) - Igor's eulogy talks about how he always smiled (90%)
    """
    )
    chain = prompt_template | llm | StrOutputParser()

    g_debug_info.documents = facts_to_inject
    g_debug_info.count_tokens = timing.token_count
    g_debug_info.question = question
    g_debug_info.model = model_name_used

    timing.start_llm()
    response = await chain.ainvoke({"question": question, "context": context})
    timing.end_llm()

    timing.start_post_llm()
    if debug:
        ic(
            f"LLM inference completed in {timing.llm_end - timing.llm_start:.2f} seconds"
        )
    timing.end_post_llm()

    timing.finish()
    if debug:
        timing.print_stats()  # Print stats if debug is on

    return response


async def iask_where_logic(
    topic: str, num_docs: int = 20, debug: bool = False, model: str = "openai"
) -> str:  # Returns formatted string
    global g_debug_info  # Allow modification
    timing = TimingStats()
    timing.start_overall_init()

    llm = get_model_for_name(model)
    model_name_used = langchain_helper.get_model_name(llm)
    timing.model_name = model_name_used
    if debug:
        ic(f"Using model: {model_name_used}")
    timing.end_overall_init()

    timing.start_db_load()
    db = get_faiss_db()
    timing.end_db_load()
    if db is None:
        raise Exception("Blog database (FAISS) not loaded.")

    timing.start_rag()
    docs_and_scores = await db.asimilarity_search_with_relevance_scores(
        topic, k=num_docs
    )
    timing.end_rag()

    if debug:
        ic(
            f"RAG retrieval completed in {timing.rag_end - timing.rag_start:.2f} seconds"
        )
        for doc, score in docs_and_scores:
            ic(doc.metadata, score)

    timing.start_doc_prep()
    facts_to_inject = [doc for doc, _ in docs_and_scores]
    context = docs_to_prompt(facts_to_inject)  # Uses fixup_markdown_path internally
    timing.token_count = num_tokens_from_string(context)
    timing.end_doc_prep()

    prompt_template = ChatPromptTemplate.from_template(
        """
You are an expert blog organization consultant. You help Igor organize his blog content effectively.
Use chain of thought reasoning to suggest where new content about a topic should be added.

Topic to add: {topic}

Here is the current layout of the blog (from back-links.json):
<blog_information>
    {backlinks}
</blog_information>

Here is the current blog structure and content for reference (from vector search):
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
class LocationRecommendation(BaseModel):
    location: str      # Where to put the content (section name/header)
    markdown_path: str # The full markdown file path (e.g. "_d/joy.md" or "_posts/something.md")
    reasoning: str     # Why this location makes sense

class BlogPlacementSuggestion(BaseModel):
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

    parser = PydanticOutputParser(pydantic_object=BlogPlacementSuggestion)
    chain = prompt_template | llm | parser

    timing.start_backlinks_load()
    # Path to back-links.json should be configurable or reliably located
    # For now, using the hardcoded path from original iwhere.py
    backlinks_file_path = (
        pathlib.Path.home() / "gits/idvorkin.github.io/back-links.json"
    )
    backlinks_content = ""
    if backlinks_file_path.exists():
        backlinks_content = backlinks_file_path.read_text()
    else:
        logger.warning(f"back-links.json not found at {backlinks_file_path}")
    timing.end_backlinks_load()

    g_debug_info.documents = facts_to_inject  # Or a subset if too large
    g_debug_info.count_tokens = timing.token_count
    g_debug_info.question = topic  # Store topic as question for debug
    g_debug_info.model = model_name_used

    timing.start_llm()
    # The PydanticOutputParser will attempt to parse the LLM output into BlogPlacementSuggestion
    parsed_result: BlogPlacementSuggestion = await chain.ainvoke(
        {"topic": topic, "context": context, "backlinks": backlinks_content}
    )
    timing.end_llm()

    timing.start_post_llm()
    if debug:
        ic(
            f"LLM inference completed in {timing.llm_end - timing.llm_start:.2f} seconds"
        )

    # Format the parsed_result into the string format expected by the original iwhere.py
    response_str = f"""
RECOMMENDED LOCATIONS:

PRIMARY LOCATION:
File Path: {parsed_result.primary_location.markdown_path}
Location: {parsed_result.primary_location.location}
Reasoning: {parsed_result.primary_location.reasoning}

ALTERNATIVE LOCATIONS:
{
        chr(10).join(
            f'''Location {i + 1}:
File Path: {loc.markdown_path}
Location: {loc.location}
Reasoning: {loc.reasoning}
'''
            for i, loc in enumerate(parsed_result.alternative_locations)
        )
    }

ADDITIONAL SUGGESTIONS:

Structuring Tips:
{chr(10).join(f"• {tip}" for tip in parsed_result.structuring_tips)}

Organization Tips:
{chr(10).join(f"• {tip}" for tip in parsed_result.organization_tips)}
"""
    timing.end_post_llm()

    timing.finish()
    if debug:
        timing.print_stats()

    return response_str


# Example of how to initialize logger if this file is run directly (e.g., for testing)
# if __name__ == '__main__':
#     logger.remove()
#     logger.add(sys.stderr, level="DEBUG")
#     logger.info("askig_logic.py loaded for testing")
# Add test calls here
