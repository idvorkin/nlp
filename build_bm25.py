#!/usr/bin/env python3
"""Quick script to build BM25 index from existing FAISS database"""

import pickle
import os
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configuration
FAISS_DB_DIR = "blog.faiss"
BM25_FILENAME = "bm25.pkl"

# Load existing FAISS DB to get documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)

# Extract all documents from FAISS
docs = list(db.docstore._dict.values())
print(f"Found {len(docs)} documents in FAISS index")

# Build BM25 index
print("Building BM25 index...")
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10  # Set default k

# Save BM25 index in the same directory as FAISS
bm25_path = os.path.join(FAISS_DB_DIR, BM25_FILENAME)
with open(bm25_path, "wb") as f:
    pickle.dump(bm25_retriever, f)

print(f"✅ BM25 index saved to {bm25_path}")
