from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def build_vectorstore(slides, embeddings):
    """
    Build FAISS vector store from labelled slides.

    Each slide dict is expected to have: id, section, text, and optionally source.
    Metadata stored per document: slide (int), section (str), source (str).
    The 'source' field distinguishes summary, raw, and image_description chunks.
    """
    docs = []
    for slide in slides:
        metadata = {
            "slide": slide["id"],
            "section": slide["section"],
            "source": slide.get("source", "summary"),
        }
        docs.append(Document(page_content=slide["text"], metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
