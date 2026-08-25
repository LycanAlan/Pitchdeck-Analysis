from pathlib import Path
from dotenv import load_dotenv
import os

from loader import load_pitchdeck_json
from embedder import get_embedder
from vectorstore import build_vectorstore
from rag_pipeline import build_rag_pipeline

load_dotenv()


def setup_rag():
    """Load data, build embeddings, and create the RAG pipeline."""
    print("🔧 Building RAG pipeline from labelled pitchdeck...")

    # Path to your JSON
    BASE_DIR = Path(__file__).resolve().parent.parent
    json_path = BASE_DIR / "preprocessing" / "outputs" / "SensonVision Jan '25-1_analysis.json"

    if not json_path.exists():
        raise FileNotFoundError(f"❌ JSON file not found: {json_path}")

    slides = load_pitchdeck_json(json_path)
    embeddings = get_embedder()
    vectorstore = build_vectorstore(slides, embeddings)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

    qa_chain = build_rag_pipeline(vectorstore, api_key)
    print(f"DEBUG: build_rag_pipeline returned {type(qa_chain)}")

    print("✅ RAG pipeline ready.\n")
    return qa_chain


def answer_query(query, qa_chain):
    """Ask a question to the RAG pipeline."""
    try:
        print(f"Type of qa_chain: {type(qa_chain)}")

        # Correct key — must match "question" in the RAG prompt
        result = qa_chain.invoke({"question": str(query)})
        print(f"\n💬 Answer:\n{result}\n")

    except Exception as e:
        print(f"❌ Error during query processing: {e}")
