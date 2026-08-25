# rag_pipeline/embedder.py
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedder():
    """Return a Hugging Face embeddings model for text vectorization."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # light & fast
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"✅ Using Hugging Face embeddings: {model_name}")
    return embeddings
