"""
rag_pipeline/retriever.py — Hybrid Retrieval with Optional Re-ranking

Implements:
  1. Dense retrieval via FAISS (semantic similarity)
  2. Sparse retrieval via BM25 (keyword/exact-match — good for proper nouns,
     figures, company names like VizSort-M, ₹11.5 Cr, etc.)
  3. Ensemble fusion of the above two using LangChain's EnsembleRetriever
  4. Optional cross-encoder re-ranking of the fused candidate set

Design rationale:
  Pitch decks are full of proper nouns, model numbers, and financial figures
  (e.g. "₹16 Cr", "gemini-2.5-flash", "M2-M20") that dense embeddings often
  represent poorly. BM25 complements dense retrieval by boosting exact-match hits.
  Cross-encoder re-ranking then reorders the fused candidates by query relevance,
  improving precision without sacrificing the recall gains from the ensemble.
"""

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Cross-encoder is optional — only imported if use_reranker=True
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_DENSE = 5    # fetch extra dense candidates before re-ranking
TOP_K_SPARSE = 5   # fetch extra sparse candidates before re-ranking
FINAL_TOP_K = 3    # number of results returned to the LLM after re-ranking


def build_hybrid_retriever(vectorstore, slides: list, use_reranker: bool = True):
    """
    Build a hybrid BM25 + FAISS ensemble retriever with optional cross-encoder re-ranking.

    Args:
        vectorstore: A FAISS vectorstore (already built from slides).
        slides: The raw list of slide dicts used to build the vectorstore.
                BM25 is built directly from this corpus.
        use_reranker: If True, adds cross-encoder re-ranking on top of the ensemble.
                      Set to False to disable and reduce latency/dependencies.

    Returns:
        A callable: fn(question: str) -> list of (Document, score) tuples.
        Score is either cosine similarity (dense) or cross-encoder score (re-ranked).
    """
    # ── 1. Dense retriever (FAISS) ─────────────────────────────────────────
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})

    # ── 2. Sparse retriever (BM25) ─────────────────────────────────────────
    bm25_docs = [
        Document(
            page_content=s["text"],
            metadata={
                "slide": s["id"],
                "section": s["section"],
                "source": s.get("source", "summary"),
            },
        )
        for s in slides
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=TOP_K_SPARSE)

    # ── 3. Ensemble fusion ─────────────────────────────────────────────────
    # weights: [dense, sparse] — give dense slightly more weight for semantic queries
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    # ── 4. Optional cross-encoder re-ranker ────────────────────────────────
    if use_reranker:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)

            def retrieve_and_rerank(question: str):
                candidates = ensemble.invoke(question)
                # Score each candidate against the query
                pairs = [(question, doc.page_content) for doc in candidates]
                scores = cross_encoder.predict(pairs)
                # Sort by score descending and trim to FINAL_TOP_K
                ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                return [(doc, float(score)) for score, doc in ranked[:FINAL_TOP_K]]

            return retrieve_and_rerank

        except ImportError:
            print("⚠️  sentence-transformers not available — falling back to ensemble without re-ranking.")

    # Fallback: ensemble without re-ranking
    def retrieve_no_rerank(question: str):
        candidates = ensemble.invoke(question)
        # Return with placeholder score of 0.0 (ensemble doesn't expose scores)
        return [(doc, 0.0) for doc in candidates[:FINAL_TOP_K]]

    return retrieve_no_rerank
