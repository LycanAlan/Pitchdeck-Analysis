from pathlib import Path
from dotenv import load_dotenv
import os

from loader import load_pitchdeck_combined, load_image_descriptions
from embedder import get_embedder
from vectorstore import build_vectorstore
from rag_pipeline import build_rag_pipeline

load_dotenv()


# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
_ANALYSIS_JSON = BASE_DIR / "preprocessing" / "outputs" / "SensonVision Jan '25-1_analysis.json"
_PARSED_JSON = BASE_DIR / "preprocessing" / "outputs" / "SensonVision Jan '25-1_parsed.json"
_IMAGE_DESC_JSON = BASE_DIR / "preprocessing" / "outputs" / "SensonVision Jan '25-1_image_descriptions.json"


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_rag(include_images: bool = False, use_reranker: bool = True):
    """
    Load data, build embeddings, and create the RAG pipeline.

    Args:
        include_images: If True, also indexes Gemini image descriptions
                        (requires preprocessing/outputs/*_image_descriptions.json to exist).
        use_reranker: If True, applies cross-encoder re-ranking on top of BM25+FAISS ensemble.
    """
    print("🔧 Building RAG pipeline from labelled pitchdeck...")

    if not _ANALYSIS_JSON.exists():
        raise FileNotFoundError(f"❌ Analysis JSON not found: {_ANALYSIS_JSON}")

    # Load summaries + raw text (Priority 3: combined indexing)
    slides = load_pitchdeck_combined(str(_ANALYSIS_JSON), str(_PARSED_JSON))
    print(f"   Loaded {len(slides)} retrievable chunks (summaries + raw text).")

    # Optionally include image descriptions (Priority 5)
    if include_images:
        image_slides = load_image_descriptions(str(_IMAGE_DESC_JSON))
        if image_slides:
            slides.extend(image_slides)
            print(f"   Added {len(image_slides)} image description chunks.")
        else:
            print("   ⚠️  No image descriptions found — run preprocessing with --with-images first.")

    embeddings = get_embedder()
    vectorstore = build_vectorstore(slides, embeddings)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

    # Pass slides list to enable BM25 hybrid retrieval (Priority 4)
    qa_fn = build_rag_pipeline(vectorstore, api_key, slides=slides, use_reranker=use_reranker)
    print("✅ RAG pipeline ready.\n")
    return qa_fn


# ─── Query ────────────────────────────────────────────────────────────────────

def answer_query(query: str, qa_fn):
    """
    Ask a question to the RAG pipeline and print the answer with source citations.

    Prints:
      - A low-confidence warning if retrieval scores are poor
      - The answer text
      - Cited slide numbers, sections, and source types (summary / raw / image_description)
    """
    try:
        result = qa_fn(str(query))

        # Low-confidence warning (Priority 2: hallucination mitigation)
        if result.get("low_confidence"):
            print("\n⚠️  LOW CONFIDENCE: The retrieved context has poor relevance to your question.")
            print("   The answer below may not be grounded in the pitch deck. Treat it with caution.\n")

        print(f"\n💬 Answer:\n{result['answer']}\n")

        # Source citations (Priority 2)
        sources = result.get("sources", [])
        if sources:
            seen_slides = set()
            citation_parts = []
            for s in sources:
                slide_id = s.get("slide")
                if slide_id not in seen_slides:
                    seen_slides.add(slide_id)
                    src_type = s.get("source", "summary")
                    section = s.get("section", "")
                    citation_parts.append(f"Slide {slide_id} [{section}] ({src_type})")
            print(f"📌 Sources: {' | '.join(citation_parts)}")

        print()

    except Exception as e:
        print(f"❌ Error during query processing: {e}")
