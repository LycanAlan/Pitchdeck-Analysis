from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_vectorstore(slides, embeddings):
    """Build FAISS vector store from labelled slides."""
    docs = []
    for slide in slides:
        metadata = {"slide": slide["id"], "section": slide["section"]}
        docs.append(Document(page_content=slide["text"], metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
