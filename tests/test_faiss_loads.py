from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.docstore.document import Document

def test_faiss_loads(tmp_path):
    docs = [Document(page_content="hello"), Document(page_content="world")]
    embeddings = FakeEmbeddings(size=3)
    index = FAISS.from_documents(docs, embeddings)
    index.save_local(str(tmp_path))
    loaded = FAISS.load_local(str(tmp_path), embeddings, allow_dangerous_deserialization=True)
    results = loaded.similarity_search("hello", k=2)
    texts = [r.page_content for r in results]
    assert "hello" in texts
