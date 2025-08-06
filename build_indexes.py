#!/usr/bin/env python3
"""Build FAISS and BM25 indexes for blog with proper chunking"""

import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from icecream import ic
from qa_blog import (
    get_blog_content,
    chunk_documents_as_md,
    chunk_documents_as_md_large,
    chunk_documents_recursive,
    dedup_chunks,
)


def build_indexes():
    """Build both FAISS and BM25 indexes with batch processing"""

    # Get all blog documents
    docs = list(get_blog_content("~/blog"))
    ic(f"Found {len(docs)} documents")

    # Chunk documents
    print("Chunking documents...")
    chunks = list(chunk_documents_as_md(docs))
    chunks += list(chunk_documents_as_md_large(docs))
    chunks += list(chunk_documents_recursive(docs))
    deduped_chunks = dedup_chunks(chunks)
    ic(f"Total chunks: {len(chunks)}, Deduped: {len(deduped_chunks)}")

    # Filter out overly large chunks (>100K chars to be safe)
    MAX_CHUNK_SIZE = 100000  # ~25K tokens
    filtered_chunks = []
    skipped = 0

    for chunk in deduped_chunks:
        if len(chunk.page_content) > MAX_CHUNK_SIZE:
            ic(
                f"Skipping large chunk from {chunk.metadata['source']}: {len(chunk.page_content)} chars"
            )
            skipped += 1
        else:
            filtered_chunks.append(chunk)

    ic(f"Filtered chunks: {len(filtered_chunks)}, Skipped: {skipped}")

    # Build FAISS index in batches
    print("Building FAISS index in batches...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    BATCH_SIZE = 50  # Process 50 documents at a time
    db = None

    for i in range(0, len(filtered_chunks), BATCH_SIZE):
        batch = filtered_chunks[i : i + BATCH_SIZE]
        print(
            f"Processing batch {i // BATCH_SIZE + 1}/{(len(filtered_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )

        if db is None:
            # Create initial FAISS index with first batch
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Add to existing index
            db.add_documents(batch)

    # Save FAISS index
    db.save_local("blog.faiss")
    print("✅ FAISS index saved to blog.faiss/")

    # Build BM25 index
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = 10

    # Save BM25 index
    with open("blog_bm25.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("✅ BM25 index saved to blog_bm25.pkl")

    print("\n✨ Both indexes built successfully!")
    return len(filtered_chunks)


if __name__ == "__main__":
    num_chunks = build_indexes()
    print(f"Total indexed chunks: {num_chunks}")
