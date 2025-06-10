#!uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "loguru",
#     "tqdm",
#     "icecream",
#     "langchain",
#     "langchain-community",
#     "langchain-core",
#     "langchain-openai",
#     "langchain-google-genai",
#     "langchain-anthropic",
#     "langchain-groq",
#     "openai",
#     "pydantic",
#     "requests",
#     "faiss-cpu",
# ]
# ///

import os
import pathlib
from typing import List  # Keep List for type hinting if any functions return List[Document] directly
from langchain.docstore.document import Document

try:
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    raise ImportError(
        "FAISS library is required. Install with `pip install faiss-cpu` or `pip install faiss-gpu`"
    ) from e

from langchain import text_splitter
from openai_wrapper import num_tokens_from_string # Assuming this is from your openai_wrapper.py
from loguru import logger
from tqdm import tqdm

import langchain_helper # For getting embeddings model
# We need to know where to build the DB. This could be passed as an argument to perform_build
# or defined here if it's considered part of the build logic's responsibility.
# For now, FAISS_INDEX_NAME and DEFAULT_FAISS_INDEX_DIR are defined here for the build process.
# askig_logic.py uses similar constants for loading the index.
# Ideally, these might come from a shared configuration file/module to avoid duplication.
FAISS_INDEX_NAME = "blog.faiss"
DEFAULT_FAISS_INDEX_DIR = FAISS_INDEX_NAME


# Directories to exclude from indexing (used by build process)
EXCLUDED_DIRS = [
    "zz-chop-logs",
    "chop-logs",
    "cursor-logs",
    "__pycache__",
    ".git",
    ".venv",
    ".cursor",
    "node_modules",
]

# Adjust to a safer size that won't exceed token limits (used by build process)
chunk_size_tokens = 2000


def should_exclude_path(path_str: str) -> bool:
    """Check if the path should be excluded from indexing."""
    path = pathlib.Path(path_str)
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            logger.debug(f"Excluding path: {path_str} (contains excluded directory: {part})")
            return True
    return False

def get_blog_content(blog_repo_path: str = "~/blog"):
    """Yields Documents from markdown files in the specified path, excluding EXCLUDED_DIRS."""
    repo_path = pathlib.Path(os.path.expanduser(blog_repo_path))
    logger.info(f"Scanning for markdown files in {repo_path}")
    logger.info(f"Excluding directories: {EXCLUDED_DIRS}")
    
    markdown_files = []
    skipped_dirs = set()
    
    for root, dirs, files in os.walk(repo_path):
        original_dirs = dirs.copy()
        # Modify dirs in-place to prevent os.walk from traversing excluded ones
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        skipped_this_level = set(original_dirs) - set(dirs)
        if skipped_this_level:
            skipped_dirs.update(skipped_this_level)
            logger.debug(f"At {root}, skipping subdirectories: {', '.join(skipped_this_level)}")
        
        for file_name in files:
            if file_name.endswith('.md'):
                full_path = pathlib.Path(root) / file_name
                # Check if the full_path itself (not just a parent dir) should be excluded
                # This is usually redundant if parent dirs are excluded, but good for direct file checks.
                if not should_exclude_path(str(full_path)):
                    markdown_files.append(full_path)
                else:
                    logger.debug(f"Excluding specific file: {full_path}")

    logger.info(f"Found {len(markdown_files)} markdown files for indexing after filtering.")
    if skipped_dirs:
        logger.info(f"Overall skipped directory names during scan: {', '.join(skipped_dirs)}")
    
    for markdown_file_path in markdown_files:
        try:
            with open(markdown_file_path, "r", encoding="utf-8") as f:
                yield Document(
                    page_content=f.read(),
                    metadata={"source": str(markdown_file_path.relative_to(repo_path))},
                )
        except Exception as e:
            logger.error(f"Error reading {markdown_file_path}: {e}")

def chunk_documents_recursive(documents: List[Document], chunk_size_cfg: int = chunk_size_tokens) -> List[Document]:
    """Split documents using recursive character splitting with token limit checks."""
    recursive_splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_cfg, chunk_overlap=chunk_size_cfg // 4
    )
    output_chunks = []
    total_skipped = 0
    for document in documents:
        try:
            doc_token_count = num_tokens_from_string(document.page_content)
            if doc_token_count > 100000:  # If document is extremely large
                logger.warning(f"Document too large for standard recursive chunking ({doc_token_count} tokens): {document.metadata.get('source', 'unknown')}. Using smaller emergency chunks.")
                temp_chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200 # Emergency smaller chunks
                ).split_text(document.page_content)
            else:
                temp_chunks = recursive_splitter.split_text(document.page_content)
                
            for chunk_text in temp_chunks:
                token_count = num_tokens_from_string(chunk_text)
                if token_count > 8000:  # OpenAI's embeddings limit per chunk
                    logger.warning(f"Recursive chunk too large after splitting: {token_count} tokens. Skipping. Source: {document.metadata.get('source', 'unknown')}")
                    total_skipped += 1
                    continue
                output_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_method": "recursive_char",
                        "source": document.metadata["source"],
                        "is_entire_document": len(temp_chunks) == 1, # True if this document was not split further by this recursive step
                    },
                ))
        except Exception as e:
            logger.error(f"Error in chunk_documents_recursive for {document.metadata.get('source', 'unknown')}: {e}")
    if total_skipped > 0:
        logger.warning(f"Skipped {total_skipped} recursive chunks that exceeded token limits after splitting.")
    return output_chunks

def chunk_documents_as_md(documents: List[Document], chunk_size_cfg: int = chunk_size_tokens) -> List[Document]:
    """Split documents using Markdown headers with token limit checks."""
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
    markdown_splitter = text_splitter.MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    output_chunks = []
    total_skipped = 0

    for document in documents:
        source_file = document.metadata.get('source', 'unknown')
        try:
            if not document.page_content or len(document.page_content) < 10:
                logger.debug(f"Document too short or empty: {source_file}. Skipping for MD chunking.")
                continue
            
            has_headers = any(header[0] in document.page_content for header in headers_to_split_on)
            
            if not has_headers:
                logger.debug(f"No markdown headers found in {source_file}. Using recursive splitting as fallback within MD chunker.")
                # Fallback to recursive for this document if no headers
                temp_recursive_chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size_cfg, chunk_overlap=chunk_size_cfg // 4
                ).split_text(document.page_content)
                
                for chunk_text in temp_recursive_chunks:
                    token_count = num_tokens_from_string(chunk_text)
                    if token_count > 8000:
                        total_skipped +=1; continue
                    output_chunks.append(Document(
                        page_content=chunk_text,
                        metadata={"source": source_file, "chunk_method": "recursive_fallback_in_md", "is_entire_document": len(temp_recursive_chunks) == 1}
                    ))
                continue # Move to next document
            
            # Proceed with Markdown splitting
            try:
                md_split_chunks = markdown_splitter.split_text(document.page_content)
            except Exception as e_md_split:
                logger.warning(f"Error splitting markdown in {source_file}: {e_md_split}. Falling back to recursive splitting for this doc.")
                temp_recursive_chunks = text_splitter.RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size_cfg, chunk_overlap=chunk_size_cfg // 4
                ).split_text(document.page_content)
                for chunk_text in temp_recursive_chunks: # Fallback recursive splitting for this document.
                    token_count = num_tokens_from_string(chunk_text)
                    if token_count > 8000: total_skipped +=1; continue
                    output_chunks.append(Document(page_content=chunk_text, metadata={"source": source_file, "chunk_method": "recursive_md_split_error", "is_entire_document": len(temp_recursive_chunks) == 1}))
                continue

            base_metadata = {"source": source_file, "chunk_method": "md_simple", "is_entire_document": False}
            for langchain_doc_chunk in md_split_chunks: # These are already Document objects from MarkdownHeaderTextSplitter
                token_count = num_tokens_from_string(langchain_doc_chunk.page_content)
                if token_count > 8000:
                    logger.warning(f"MD chunk too large: {token_count} tokens from {source_file}. Skipping.")
                    total_skipped += 1
                    continue
                # Merge metadata, ensuring source and chunk_method from base_metadata take precedence if needed
                final_metadata = {**langchain_doc_chunk.metadata, **base_metadata}
                output_chunks.append(Document(page_content=langchain_doc_chunk.page_content, metadata=final_metadata))
        except Exception as e:
            logger.error(f"Outer error in chunk_documents_as_md for {source_file}: {e}")
    
    if total_skipped > 0:
        logger.warning(f"Skipped {total_skipped} MD (or fallback recursive) chunks that exceeded token limits.")
    return output_chunks

def chunk_documents_as_md_large(documents: List[Document], chunk_size_cfg: int = chunk_size_tokens) -> List[Document]:
    """Create larger chunks from markdown documents by merging smaller header-split chunks up to token limits."""
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
    markdown_splitter = text_splitter.MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    output_merged_chunks = []
    total_skipped_individual_chunks = 0
    total_merged_docs_processed = 0

    for document in documents:
        source_file = document.metadata.get('source', 'unknown')
        try:
            if not document.page_content or len(document.page_content) < 10:
                logger.debug(f"Document too short or empty: {source_file}. Skipping for MD large chunking.")
                continue
            
            doc_token_count = num_tokens_from_string(document.page_content)
            if doc_token_count > 100000: # Heuristic: very large docs might be problematic for this merging strategy
                logger.warning(f"Document too large for MD large chunking ({doc_token_count} tokens): {source_file}. Skipping this doc for this strategy.")
                continue

            if not any(header[0] in document.page_content for header in headers_to_split_on):
                logger.debug(f"No markdown headers in {source_file}. Skipping for MD large chunker (expects headers).")
                continue
            
            try:
                # Initial split by headers. These are Langchain Document objects.
                individual_md_chunks = markdown_splitter.split_text(document.page_content) 
            except Exception as e_md_split:
                logger.warning(f"Error during initial MD splitting in md_large for {source_file}: {e_md_split}. Skipping this doc for this strategy.")
                continue

            if not individual_md_chunks:
                logger.debug(f"No chunks produced by MD splitter for {source_file} in md_large. Skipping.")
                continue

            current_merged_content = ""
            current_merged_metadata = {} # For headers from the first chunk in a merged group
            MAX_TOKENS_PER_EMBED_CHUNK = 8000 # OpenAI limit
            TARGET_MERGE_SIZE = int(MAX_TOKENS_PER_EMBED_CHUNK * 0.85) # Target slightly less than max

            temp_buffer_for_merging = []

            for md_chunk_doc in individual_md_chunks:
                chunk_text = md_chunk_doc.page_content
                chunk_tokens = num_tokens_from_string(chunk_text)

                if chunk_tokens > MAX_TOKENS_PER_EMBED_CHUNK:
                    logger.warning(f"Single MD sub-chunk from {source_file} too large ({chunk_tokens} tokens) even before merging. Skipping this sub-chunk.")
                    total_skipped_individual_chunks += 1
                    continue
                
                # If current buffer + new chunk exceeds target, yield the buffer.
                current_buffer_tokens = num_tokens_from_string("".join(c.page_content for c in temp_buffer_for_merging))
                if temp_buffer_for_merging and (current_buffer_tokens + chunk_tokens > TARGET_MERGE_SIZE):
                    page_content_to_yield = "\n\n---\n\n".join(c.page_content for c in temp_buffer_for_merging)
                    # Metadata for the merged chunk: from the first chunk in the buffer + source and method
                    merged_metadata = {
                        **temp_buffer_for_merging[0].metadata, 
                        "source": source_file, 
                        "chunk_method": "md_merge",
                        "is_entire_document": False # It's a merged chunk, not the whole original doc unless it was tiny
                    }
                    output_merged_chunks.append(Document(page_content=page_content_to_yield, metadata=merged_metadata))
                    temp_buffer_for_merging = [] # Reset buffer

                temp_buffer_for_merging.append(md_chunk_doc) # Add current chunk to buffer

            # Yield any remaining content in the buffer
            if temp_buffer_for_merging:
                page_content_to_yield = "\n\n---\n\n".join(c.page_content for c in temp_buffer_for_merging)
                merged_metadata = {
                     **temp_buffer_for_merging[0].metadata,
                    "source": source_file, 
                    "chunk_method": "md_merge",
                     # If the original doc was small enough to fit entirely into one merged chunk
                    "is_entire_document": len(individual_md_chunks) == len(temp_buffer_for_merging) 
                }
                output_merged_chunks.append(Document(page_content=page_content_to_yield, metadata=merged_metadata))
            
            total_merged_docs_processed +=1

        except Exception as e:
            logger.error(f"Outer error in chunk_documents_as_md_large for {source_file}: {e}")
    
    logger.info(f"MD large chunking: processed {total_merged_docs_processed} documents suitable for this strategy.")
    if total_skipped_individual_chunks > 0:
        logger.warning(f"MD large chunking: skipped {total_skipped_individual_chunks} individual sub-chunks that were too large.")
    return output_merged_chunks


def dedup_chunks(chunks: List[Document]) -> List[Document]:
    """Remove duplicate chunks. Prioritizes keeping chunks marked as 'is_entire_document' if content is similar."""
    unique_by_content_hash = {}
    skipped_due_to_metadata = 0

    for chunk in chunks:
        if not chunk.metadata or "source" not in chunk.metadata or "is_entire_document" not in chunk.metadata:
            logger.warning(f"Chunk missing essential metadata (source/is_entire_document): {chunk.metadata.get('source', 'Unknown source')}. Skipping.")
            skipped_due_to_metadata += 1
            continue
        
        content = chunk.page_content.strip()
        if not content:
            logger.debug(f"Skipping empty chunk from {chunk.metadata['source']}.")
            continue
            
        # Simple hash for content deduplication
        content_hash = hash(content[:1000] + content[-1000:] if len(content) > 2000 else content)
        
        existing_chunk = unique_by_content_hash.get(content_hash)
        if existing_chunk:
            # If new chunk is entire doc and old one wasn't, prefer new one.
            if chunk.metadata["is_entire_document"] and not existing_chunk.metadata["is_entire_document"]:
                unique_by_content_hash[content_hash] = chunk
            # If both are/aren't entire doc, or if old one is entire and new one isn't, keep existing.
            # (This implicitly prefers the first one seen if metadata['is_entire_document'] is the same)
        else:
            unique_by_content_hash[content_hash] = chunk
            
    final_unique_chunks = list(unique_by_content_hash.values())
    num_deduplicated = len(chunks) - len(final_unique_chunks) - skipped_due_to_metadata
    
    logger.info(f"Deduplication: Input {len(chunks)} -> Unique {len(final_unique_chunks)} chunks. ({num_deduplicated} content duplicates removed).")
    if skipped_due_to_metadata > 0:
        logger.warning(f"Deduplication: Skipped {skipped_due_to_metadata} chunks due to missing metadata.")
    return final_unique_chunks


def process_chunks_in_batches(chunks: List[Document], batch_size: int = 50):
    """Yields chunks in smaller batches."""
    total_chunks = len(chunks)
    if total_chunks == 0:
        logger.info("No chunks to process in batches.")
        return # Yield nothing if no chunks
    
    # Adjust batch size if it's too large (OpenAI has a 300k token limit per request)
    # A safer batch size of 100 should keep us under the limit in most cases
    if batch_size > 100:
        logger.warning(f"Requested batch size {batch_size} may exceed token limits. Reducing to 100.")
        batch_size = 100
        
    logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
    for i in tqdm(range(0, total_chunks, batch_size), desc="Batching chunks for embedding"):
        yield chunks[i:i+batch_size]

def perform_build(
    blog_repo_path: str, 
    db_persist_directory: str, # Explicitly pass where to save the DB
    batch_size_for_embedding: int
):
    """Main logic to build the vector database."""
    logger.info(f"Starting database build from blog path: {blog_repo_path}")
    logger.info(f"Database will be persisted to: {db_persist_directory}")

    # Initialize embeddings model
    # This assumes langchain_helper.get_embeddings_model() exists and returns the correct model
    # e.g., return OpenAIEmbeddings(model="text-embedding-3-large")
    try:
        embeddings_model = langchain_helper.get_embeddings_model()
        if embeddings_model is None: # Check if the helper could fail and return None
             raise ValueError("langchain_helper.get_embeddings_model() returned None")
    except Exception as e:
        logger.error(f"Failed to get embeddings model from langchain_helper: {e}. Cannot proceed with build.")
        raise
        
    raw_documents = list(get_blog_content(blog_repo_path))
    logger.info(f"Loaded {len(raw_documents)} raw documents from source.")

    if not raw_documents:
        logger.warning("No documents found to process. Build will result in an empty database.")
        # Create the directory anyway, so the CLI doesn't fail later if it expects it.
        os.makedirs(db_persist_directory, exist_ok=True)
        logger.info(f"Empty database initialized at {db_persist_directory}")
        return

    all_chunks = []
    logger.info("Starting document chunking process...")
    
    # Strategy 1: Markdown Chunks (with recursive fallback for docs without headers)
    md_chunks = chunk_documents_as_md(raw_documents, chunk_size_cfg=chunk_size_tokens)
    all_chunks.extend(md_chunks)
    logger.info(f"Generated {len(md_chunks)} chunks using Markdown header splitting (with fallbacks). Total: {len(all_chunks)}")
    
    # Strategy 2: Larger Merged Markdown Chunks (for docs with headers)
    md_large_chunks = chunk_documents_as_md_large(raw_documents, chunk_size_cfg=chunk_size_tokens)
    all_chunks.extend(md_large_chunks)
    logger.info(f"Generated {len(md_large_chunks)} using merged Markdown strategy. Total: {len(all_chunks)}")

    # Strategy 3: Pure Recursive Chunks (as a general catch-all or for non-MD heavy content)
    recursive_chunks_list = chunk_documents_recursive(raw_documents, chunk_size_cfg=chunk_size_tokens)
    all_chunks.extend(recursive_chunks_list)
    logger.info(f"Generated {len(recursive_chunks_list)} using general recursive splitting. Grand Total: {len(all_chunks)}")
    
    if not all_chunks:
        logger.warning("No chunks were generated from the documents. Build will result in an empty database.")
        os.makedirs(db_persist_directory, exist_ok=True)
        logger.info(f"Empty database initialized at {db_persist_directory}")
        return

    deduplicated_chunks = dedup_chunks(all_chunks)
    logger.info(f"Total chunks after deduplication: {len(deduplicated_chunks)}")

    if not deduplicated_chunks:
        logger.warning("All chunks were duplicates or empty. Build will result in an empty database.")
        os.makedirs(db_persist_directory, exist_ok=True)
        logger.info(f"Empty database initialized at {db_persist_directory}")
        return

    os.makedirs(db_persist_directory, exist_ok=True)
    
    db = None
    logger.info(
        f"Embedding and ingesting chunks into FAISS index at {db_persist_directory} in batches of {batch_size_for_embedding}"
    )
    
    batch_generator = process_chunks_in_batches(deduplicated_chunks, batch_size=batch_size_for_embedding)

    try:
        for i, batch_of_chunks in enumerate(batch_generator):
            if not batch_of_chunks: # Should not happen if process_chunks_in_batches handles empty list
                logger.warning(f"Batch {i+1} is empty. Skipping.")
                continue

            logger.info(f"Processing batch {i+1} with {len(batch_of_chunks)} chunks for embedding.")
            if i == 0:
                logger.info("Creating new FAISS index with the first batch.")
                db = FAISS.from_documents(batch_of_chunks, embeddings_model)
            else:
                if db is None:  # Should not happen after first batch if db was created.
                    logger.warning(
                        "FAISS index instance is None after the first batch. Attempting to reconnect."
                    )
                    db = FAISS.load_local(
                        db_persist_directory,
                        embeddings_model,
                        allow_dangerous_deserialization=True,
                    )
                logger.info(f"Adding batch {i+1} to existing FAISS index.")
                db.add_documents(batch_of_chunks)
            
            logger.info(f"Batch {i+1} successfully processed and added to DB.")
            # Persist is usually handled automatically by FAISS on add,
            # but explicit save can be called if issues arise.
            # if db: db.save_local(db_persist_directory)

    except Exception as e:
        logger.error(f"An error occurred during database building (embedding/ingestion phase): {e}")
        logger.exception("Traceback for build error:") # Logs full traceback
        # Depending on error, db might be partially created.
        raise # Re-throw to indicate build failure

    if db is not None:
        db.save_local(db_persist_directory)
    logger.info(
        f"Successfully built and persisted the blog index at {db_persist_directory}"
    )
    # The CLI part in iwhere.py will print the final user-facing message.

# Example of how to initialize logger if this file is run directly (e.g., for testing)
# if __name__ == '__main__':
#     logger.remove()
#     logger.add(sys.stderr, level="DEBUG")
#     logger.info("askig_build_logic.py loaded for testing")
#     # Example test call (ensure DB dir exists or is handled):
#     # perform_build(blog_repo_path="~/blog", db_persist_directory="./test_blog.faiss", batch_size_for_embedding=5)
