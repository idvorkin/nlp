import time
import sys
from typing import List, Optional
from pydantic import BaseModel
from langchain.docstore.document import Document # For DebugInfo.documents

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
        self.overall_init_start = 0
        self.overall_init_end = 0
        self.db_load_start = 0
        self.db_load_end = 0
        self.doc_prep_start = 0
        self.doc_prep_end = 0
        self.backlinks_load_start = 0  # Specific to iask_where
        self.backlinks_load_end = 0    # Specific to iask_where
        self.post_llm_start = 0
        self.post_llm_end = 0
        
    def start_rag(self): self.rag_start = time.time()
    def end_rag(self): self.rag_end = time.time()
    def start_llm(self): self.llm_start = time.time()
    def end_llm(self): self.llm_end = time.time()
    def finish(self): self.end_time = time.time()
    def start_overall_init(self): self.overall_init_start = time.time()
    def end_overall_init(self): self.overall_init_end = time.time()
    def start_db_load(self): self.db_load_start = time.time()
    def end_db_load(self): self.db_load_end = time.time()
    def start_doc_prep(self): self.doc_prep_start = time.time()
    def end_doc_prep(self): self.doc_prep_end = time.time()
    def start_backlinks_load(self): self.backlinks_load_start = time.time() # For iask_where
    def end_backlinks_load(self): self.backlinks_load_end = time.time()   # For iask_where
    def start_post_llm(self): self.post_llm_start = time.time()
    def end_post_llm(self): self.post_llm_end = time.time()
        
    def print_stats(self):
        """Print timing statistics to stderr."""
        overall_init_time = self.overall_init_end - self.overall_init_start if self.overall_init_end > 0 else 0
        db_load_time = self.db_load_end - self.db_load_start if self.db_load_end > 0 else 0
        rag_time = self.rag_end - self.rag_start if self.rag_end > 0 else 0
        doc_prep_time = self.doc_prep_end - self.doc_prep_start if self.doc_prep_end > 0 else 0
        backlinks_load_time = self.backlinks_load_end - self.backlinks_load_start if self.backlinks_load_end > 0 else 0
        llm_time = self.llm_end - self.llm_start if self.llm_end > 0 else 0
        post_llm_time = self.post_llm_end - self.post_llm_start if self.post_llm_end > 0 else 0
        total_time = self.end_time - self.start_time
        
        measured_sum = overall_init_time + db_load_time + rag_time + doc_prep_time + backlinks_load_time + llm_time + post_llm_time
        unaccounted_time = total_time - measured_sum

        stats_str = f"""
=== Performance Statistics ===
Model: {self.model_name}
Overall Initialization: {overall_init_time:.2f}s
Database Loading: {db_load_time:.2f}s
RAG retrieval time: {rag_time:.2f}s
Document Preparation: {doc_prep_time:.2f}s
"""
        if backlinks_load_time > 0: # Only print if relevant for iask_where
            stats_str += f"Backlinks File Loading: {backlinks_load_time:.2f}s\n"
        stats_str += f"""LLM inference time: {llm_time:.2f}s
Post-LLM Processing: {post_llm_time:.2f}s
----------------------------
Sum of Measured Steps: {measured_sum:.2f}s
Total processing time: {total_time:.2f}s
Unaccounted time: {unaccounted_time:.2f}s
Token count: {self.token_count}
============================
"""
        print(stats_str, file=sys.stderr) 