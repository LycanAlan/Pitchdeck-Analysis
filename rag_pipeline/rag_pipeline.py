# rag_pipeline/rag_pipeline.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import build_hybrid_retriever

# Similarity score threshold for low-confidence warning.
# Cross-encoder scores are in range [-inf, +inf]; scores below this threshold
# indicate poor relevance. For ensemble fallback (score=0.0), we skip the check.
LOW_CONFIDENCE_THRESHOLD = 0.0


def build_rag_pipeline(vectorstore, api_key: str, slides: list = None, use_reranker: bool = True):
    """
    Hybrid RAG: BM25 + FAISS ensemble retrieval + optional cross-encoder re-ranking
    + Gemini reasoning LLM.

    Args:
        vectorstore: FAISS vectorstore (built from slides).
        api_key: Google Gemini API key.
        slides: Raw slide list (needed for BM25). If None, falls back to dense-only.
        use_reranker: Whether to apply cross-encoder re-ranking.

    Returns a callable fn(question: str) -> dict:
        {
          "answer": str,
          "sources": list of {slide, section, source, score},
          "low_confidence": bool
        }
    """
    # Build the hybrid retriever (falls back to dense-only if slides=None)
    if slides is not None:
        retrieve_fn = build_hybrid_retriever(vectorstore, slides, use_reranker=use_reranker)
    else:
        # Fallback: pure dense retrieval with scores
        def retrieve_fn(question: str):
            return vectorstore.similarity_search_with_score(question, k=3)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on a startup pitch deck.
Use the provided context carefully to craft an insightful and accurate answer.

Context:
{context}

Question:
{question}

Helpful, concise answer:
""")

    answer_chain = prompt | llm | StrOutputParser()

    def run(question: str) -> dict:
        docs_with_scores = retrieve_fn(question)

        context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
        answer = answer_chain.invoke({"context": context, "question": question})

        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "slide": doc.metadata.get("slide"),
                "section": doc.metadata.get("section", ""),
                "source": doc.metadata.get("source", "summary"),
                "score": round(float(score), 4),
            })

        # Low confidence: cross-encoder scores below threshold, or all scores are 0.0 (ensemble fallback)
        top_score = docs_with_scores[0][1] if docs_with_scores else None
        low_confidence = (
            top_score is not None
            and top_score != 0.0  # 0.0 = ensemble fallback, skip check
            and float(top_score) < LOW_CONFIDENCE_THRESHOLD
        )

        return {
            "answer": answer,
            "sources": sources,
            "low_confidence": low_confidence,
        }

    return run
