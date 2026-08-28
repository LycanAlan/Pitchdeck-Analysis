# -*- coding: utf-8 -*-
"""
eval.py -- Retrieval + Answer Quality Evaluation for Pitchdeck-Analysis RAG

Usage:
    python rag_pipeline/eval.py                   # Full eval (hit rate + Gemini judge)
    python rag_pipeline/eval.py --skip-judge      # Hit rate only (no Gemini API calls)
    python rag_pipeline/eval.py --output results.csv

Judge Rubric (printed below, used verbatim in the Gemini judge prompt):
    - correct:           The answer fully and accurately addresses the question.
    - partially_correct: The answer is on the right topic but is incomplete or slightly inaccurate.
    - incorrect:         The answer is factually wrong or addresses the wrong topic entirely.
    - hallucinated:      The answer contains plausible-sounding but fabricated information not in the deck.

Retrieval Hit Rate:
    For each question, we check whether any of the ground-truth slide numbers appear
    in the top-k documents retrieved by FAISS. k=3 by default (matching the pipeline).
"""

import sys
import os
import csv
import json
import argparse
from pathlib import Path

# Ensure rag_pipeline package is importable when run from project root or within the package dir
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
import google.genai as genai

from loader import load_pitchdeck_json
from embedder import get_embedder
from vectorstore import build_vectorstore

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_JSON = BASE_DIR / "preprocessing" / "outputs" / "SensonVision Jan '25-1_analysis.json"
EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "eval_results.csv"

TOP_K = 3  # number of chunks retrieved per question

JUDGE_RUBRIC = """
You are an objective evaluator. Score the generated answer against the expected answer
using EXACTLY one of these labels:

  correct           — Fully and accurately answers the question.
  partially_correct — On the right topic but incomplete or slightly inaccurate.
  incorrect         — Factually wrong or addresses the wrong topic entirely.
  hallucinated      — Contains plausible-sounding but fabricated information not from the deck.

Reply with ONLY the label (one word or compound word from the list above) and nothing else.
"""


# ─── Setup ────────────────────────────────────────────────────────────────────

def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_retriever(analysis_path):
    """Load data, embed, build FAISS -- mirrors setup_rag() but returns the vectorstore directly."""
    print("[EVAL] Building vectorstore for evaluation...")
    slides = load_pitchdeck_json(str(analysis_path))
    embedder = get_embedder()
    vectorstore = build_vectorstore(slides, embedder)
    print(f"[EVAL] Indexed {len(slides)} slides.\n")
    return vectorstore


def retrieve_with_scores(vectorstore, question: str, k: int = TOP_K):
    """Returns list of (Document, score) tuples from FAISS."""
    return vectorstore.similarity_search_with_score(question, k=k)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def check_hit(retrieved_docs, ground_truth_slides):
    """Retrieval hit: did any ground-truth slide appear in the retrieved docs?"""
    retrieved_slide_ids = set()
    for doc, _score in retrieved_docs:
        slide_id = doc.metadata.get("slide")
        if slide_id is not None:
            retrieved_slide_ids.add(int(slide_id))
    return bool(retrieved_slide_ids.intersection(set(ground_truth_slides)))


def judge_answer(question: str, expected: str, generated: str, api_key: str) -> str:
    """Ask Gemini to score the generated answer against expected. Returns label string."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
{JUDGE_RUBRIC.strip()}

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Your label:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        label = response.text.strip().lower().replace(" ", "_")
        valid_labels = {"correct", "partially_correct", "incorrect", "hallucinated"}
        return label if label in valid_labels else "unknown"
    except Exception as e:
        print(f"  [WARN] Judge API error: {e}")
        return "judge_error"


def generate_answer(vectorstore, question: str, api_key: str) -> str:
    """Run full RAG chain for a question and return the answer string."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    docs_with_scores = retrieve_with_scores(vectorstore, question)
    context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # use flash for eval to reduce cost
        google_api_key=api_key,
        temperature=0.0,
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

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval(skip_judge: bool, output_path: Path):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not skip_judge:
        raise ValueError("GEMINI_API_KEY not set. Use --skip-judge to run without Gemini.")

    eval_set = load_eval_set()
    vectorstore = build_retriever(ANALYSIS_JSON)

    results = []
    hits = 0
    judge_counts = {"correct": 0, "partially_correct": 0, "incorrect": 0, "hallucinated": 0, "unknown": 0, "judge_error": 0}

    print("=" * 70)
    print(f"{'ID':>3}  {'Hit':>5}  {'Score':>18}  Question")
    print("-" * 70)

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected = item["expected_answer"]
        gt_slides = item["ground_truth_slides"]

        # Retrieval
        retrieved = retrieve_with_scores(vectorstore, question)
        hit = check_hit(retrieved, gt_slides)
        if hit:
            hits += 1

        retrieved_slides = [int(doc.metadata.get("slide", -1)) for doc, _ in retrieved]
        top_score = retrieved[0][1] if retrieved else None

        # Answer generation + judging
        if skip_judge:
            generated = "(skipped)"
            judge_label = "skipped"
        else:
            try:
                generated = generate_answer(vectorstore, question, api_key)
                judge_label = judge_answer(question, expected, generated, api_key)
                judge_counts[judge_label] = judge_counts.get(judge_label, 0) + 1
            except Exception as e:
                generated = f"ERROR: {e}"
                judge_label = "error"

        results.append({
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "ground_truth_slides": gt_slides,
            "retrieved_slides": retrieved_slides,
            "top_similarity_score": round(float(top_score), 4) if top_score else None,
            "hit": hit,
            "judge_label": judge_label,
        })

        hit_str = "HIT" if hit else "MISS"
        q_short = question[:45] + "..." if len(question) > 45 else question
        print(f"{qid:>3}  {hit_str:>5}  {judge_label:>18}  {q_short}")

    total = len(eval_set)
    hit_rate = hits / total * 100

    # --- Summary --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions   : {total}")
    print(f"Retrieval hit rate: {hits}/{total} = {hit_rate:.1f}%  (top-{TOP_K})")
    if not skip_judge:
        print(f"\nAnswer quality (Gemini judge):")
        for label, count in judge_counts.items():
            if count > 0:
                pct = count / total * 100
                print(f"  {label:<20}: {count:>3} ({pct:.1f}%)")
    print("=" * 70)

    # --- CSV Output -----------------------------------------------------------
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results saved to: {output_path}")

    return hit_rate, judge_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline retrieval and answer quality.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip Gemini judge (hit rate only, no API calls for judging)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path for CSV output file")
    args = parser.parse_args()

    run_eval(skip_judge=args.skip_judge, output_path=Path(args.output))
