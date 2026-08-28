# PitchLens

Multimodal retrieval-augmented question answering over startup pitch decks, built around a measured claim rather than a demo.

**The problem this project exists for:** of 19 real pitch decks, **only 4 (21%) have a usable PDF text layer**. The other 15 are design exports where every slide is a flat image and `PyMuPDF.get_text()` returns an empty string. A conventional "chat with your PDF" pipeline silently ingests *nothing* for Airbnb, Uber, Dropbox, Facebook, LinkedIn, Square, Coinbase, Front, Mixpanel, Monzo, Revolut, Brex, Dwolla, Transferwise and Oscar Health.

PitchLens routes **per page**, not per deck: the text layer where it exists, a vision model where it does not. Then it measures every stage.

---

## Results

All numbers are reproducible with the scripts in [`scripts/`](scripts/). The retrieval and latency measurements run **fully offline** and need no API key.

### 1. Extraction coverage — the headline

| extractor | decks usable | % of decks | pages readable |
| :--- | :--- | ---: | ---: |
| text-layer only (the conventional pipeline) | 4/19 | 21.1% | 153 |
| **adaptive (text layer + vision)** | **19/19** | **100%** | **440** |

**4.75× more decks and 2.9× more pages become readable.** 287 of 440 pages (65%) have no text layer at all.

`python scripts/measure_ingestion.py` — pure PyMuPDF, no API calls.

### 2. Retrieval ablation

72 human-curated questions with slide-level ground truth, over 5 ingested decks / 212 chunks. Unanswerable questions are excluded from ranking metrics and counted separately.

| retriever | recall@3 | recall@5 | MRR | nDCG@5 | p50 latency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dense (MiniLM + FAISS) | 0.777 | 0.862 | 0.743 | 0.764 | 6.8 ms |
| bm25 | 0.862 | 0.923 | 0.810 | 0.830 | 0.3 ms |
| hybrid (RRF fusion) | 0.908 | 0.962 | 0.901 | 0.903 | 7.7 ms |
| dense + cross-encoder | 0.923 | 0.954 | 0.913 | 0.915 | 292 ms |
| **hybrid + cross-encoder** | **0.954** | **0.969** | **0.917** | **0.924** | 321 ms |

**Recall@3 rises from 0.777 to 0.954 (+17.7 points, +22.8% relative) and MRR from 0.743 to 0.917 (+23.4%).**

Two findings worth stating plainly:

- **BM25 alone beats dense embeddings on this corpus** (recall@3 0.862 vs 0.777). Pitch decks are dense with proper nouns and figures — `₹11.5 Cr`, `VizSort-M`, `$8.9B`, `Q4 2024` — precisely where lexical matching wins and a 384-dim sentence embedding blurs. Fusing the two beats either alone.
- **Re-ranking costs 42× the latency for +5 points of recall@3** (7.7 ms → 321 ms). Whether that trade is worth it is a product decision, and the table is what makes it a decision rather than a guess.

`python scripts/measure_retrieval.py` — offline.

### 3. Index persistence and latency

| chunks | cold build | warm load | speedup |
| ---: | ---: | ---: | :--- |
| 212 | 3.89 s | 0.036 s | **109×** |

The original implementation rebuilt the FAISS index on every process start and never persisted it. The index is now fingerprinted over the embedding-model id plus every chunk, so it rebuilds when the corpus or the model actually changes and loads from disk otherwise.

`python scripts/measure_latency.py` — offline.

### 4. Answer quality — preliminary

| system | correct | faithfulness | hallucination | mean latency |
| :--- | ---: | ---: | ---: | ---: |
| baseline (one-shot RAG) | 90.9% | 100% | 0% | 2.9 s |
| corrective agent | 100%* | 100% | 0% | 3.6 s |

> **Not yet resume-grade.** n=12, and the free-tier quota exhausted mid-run: the baseline completed 11 of 12 questions and the corrective agent only 7 (\*hence its 100%). The judge also fell back from the `gemini-3.1-pro` tier into flash, so the "stronger, different judge family" property did not hold for this run. Re-run with quota available:
> `python scripts/measure_answers.py`

---

## Architecture

```
pitchlens/
  config.py            Every tunable, once. Model chains, k values, paths.
  domain.py            Chunk, ScoredChunk, Slide, DeckDocument, Answer, EvalQuestion.
  llm.py               The single gateway for every model call.
  ingest/
    extractor.py       TextLayer / Vision / Adaptive page extractors.
    analyzer.py        Structured slide summarisation.
    pipeline.py        PDF -> DeckDocument, resumable.
  index/
    embedder.py        Cached sentence-transformers singleton.
    store.py           FAISS index with fingerprinted persistence.
  retrieval/
    base.py            Retriever ABC + decorator base.
    dense.py sparse.py hybrid.py rerank.py factory.py
  generation/
    base.py            RAGPipeline: the shared retrieve+generate contract.
    prompts.py         Every prompt in the system, once.
    answerer.py        One-shot arm.
  agent/
    corrective.py      LangGraph retrieve -> grade -> rewrite -> generate.
  evaluation/
    metrics.py         Pure ranking metrics, dependency-free.
    judge.py harness.py reporting.py
api/main.py            FastAPI service; expensive objects built once at startup.
app.py                 Streamlit thin client over the API.
scripts/               Standalone, independently runnable measurements.
```

Three design decisions carry most of the weight:

**`Retriever` is one interface.** Dense, sparse, fused and re-ranked strategies all satisfy it, and `RerankDecorator` wraps *any* retriever rather than subclassing each combination. The evaluation harness holds a `Retriever` and never learns which one it has — which is why the ablation table is a single loop over `RetrieverFactory.MODES` and adding a strategy adds zero branches anywhere else.

**`RAGPipeline` is one interface.** `Answerer` and `CorrectiveRAGAgent` share `answer(question) -> Answer` and a `name`, so the harness and the API swap arms without a conditional.

**Every model call goes through `GeminiClient`.** Free-tier quota is granted *per model per day*, and the endpoint returns 503 and 404 unpredictably, so the client walks a chain of models and permanently drops one that 429s or 404s. Chain-walking is not just failover here — it is how the pipeline gets a usable budget at all.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # add GEMINI_API_KEY

python scripts/fetch_corpus.py                       # 18 public decks
python scripts/measure_ingestion.py                  # coverage — no API key needed
python scripts/ingest_corpus.py                      # PDF -> DeckDocument (uses API)
python scripts/measure_retrieval.py                  # the ablation — no API key needed
python scripts/measure_latency.py                    # persistence + latency — no API key
python scripts/measure_answers.py                    # answer quality (uses API)
python scripts/run_all_measurements.py               # everything -> results/RESULTS.md
```

Serve it:

```bash
uvicorn api.main:app --reload      # http://localhost:8000/docs
streamlit run app.py               # thin client over the API
docker compose up                  # both
```

`scripts/ingest_corpus.py --text-only` disables vision entirely — that is the baseline extractor from the coverage table, runnable as a real pipeline rather than a hypothetical.

---

## Method notes

- **Ground truth is human-curated.** The 72 questions in `data/eval/eval_set.json` were written by reading the ingested transcripts, with gold slide numbers verified against the corpus. `scripts/build_eval_set.py` generates candidates with an LLM and drops any whose gold slides do not exist, but the shipped set is hand-checked.
- **The judge is a different, stronger model family than the generator** (`gemini-3.1-pro` vs `gemini-3-flash`) so the system does not grade its own output. When the pro tier is quota-exhausted the client falls back into flash and the run's LLM-usage table records it.
- **Deduping before scoring.** A slide is indexed as both a transcript chunk and a summary chunk, so a retriever legitimately returns it twice; identifiers are deduped preserving rank order before any metric reads a position, or precision and nDCG would both be inflated.
- **Corpus caveat.** Sections 2–4 are measured over the 5 decks ingested so far (212 chunks), not all 19. Section 1 covers all 19 because it is a pure property of the PDFs. Ingesting the remaining 14 decks needs ~287 vision calls, which exceeds the free tier's 20-requests-per-model-per-day.

## Corpus

18 public decks from the [`skyforclouds/pitch-deckz`](https://huggingface.co/datasets/skyforclouds/pitch-deckz) dataset (Airbnb, Uber, Dropbox, Facebook, LinkedIn, Coinbase, Square, Shopify, WeWork, Monzo, Revolut, Brex, Front, Mixpanel, Transferwise, Oscar Health, Dwolla, Nium) plus one private deck.
