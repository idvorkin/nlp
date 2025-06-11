import os
import tempfile
import shutil
import pytest
from unittest import mock
from askig_build_logic import BlogContentManager, VectorDBManager
from langchain_core.documents import Document

class DummyEmbeddings:
    def embed_documents(self, docs):
        return [[0.0] * 10 for _ in docs]
    def embed_query(self, doc):
        return [0.0] * 10

def test_blog_content_manager_lists_and_loads(tmp_path):
    # Setup: create a fake blog directory with allowed top-level dirs
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()
    d_dir = blog_dir / "_d"
    d_dir.mkdir()
    (d_dir / "file1.md").write_text("# Title\nContent1")
    (d_dir / "file2.md").write_text("# Title\nContent2")
    (d_dir / "skip.txt").write_text("Not markdown")
    subdir = d_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.md").write_text("# Title\nContent3")

    mgr = BlogContentManager(str(blog_dir))
    files = list(mgr.list_markdown_files())
    assert len(files) == 3
    assert all(f.endswith(".md") for f in files)

    docs = list(mgr.load_documents())
    assert len(docs) == 3
    assert all("# Title" in d.page_content for d in docs)

@mock.patch("langchain_community.vectorstores.FAISS", autospec=True)
def test_vector_db_manager_create_and_save(MockFAISS):
    dummy_embeddings = DummyEmbeddings()
    db_dir = tempfile.mkdtemp()
    try:
        mgr = VectorDBManager(db_persist_directory=db_dir, embeddings_model=dummy_embeddings)
        docs = [Document(page_content="foo", metadata={"id": "test-id"})]
        # Mock FAISS.from_documents and .save_local
        instance = MockFAISS.from_documents.return_value
        mgr.create_db(docs)
        MockFAISS.from_documents.assert_called_once_with(docs, dummy_embeddings)
        mgr.save_db()
        instance.save_local.assert_called_once_with(db_dir)
    finally:
        shutil.rmtree(db_dir) 