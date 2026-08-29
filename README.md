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

72 human-curated questions with slide-level ground truth, over the full corpus: 18 ingested decks / 730 chunks. Unanswerable questions carry no gold slides, so they are excluded from ranking metrics and counted separately (65 scored, 7 skipped).

| retriever | recall@3 | recall@5 | MRR | nDCG@5 |
| :--- | ---: | ---: | ---: | ---: |
| dense (MiniLM + FAISS) | 0.762 | 0.846 | 0.713 | 0.733 |
| bm25 | 0.800 | 0.892 | 0.772 | 0.791 |
| hybrid (RRF fusion) | 0.854 | 0.885 | 0.806 | 0.811 |
| dense + cross-encoder | 0.923 | 0.946 | 0.901 | 0.906 |
| **hybrid + cross-encoder** | **0.938** | 0.938 | **0.905** | 0.904 |

**Recall@3 rises from 0.762 to 0.938 (+17.6 points, +23.2% relative) and MRR from 0.713 to 0.905 (+26.9%).**

Ranking metrics are identical on the PyTorch and ONNX backends (embeddings match to float32 precision), and identical with and without the re-ranker's input truncation — both verified at a fixed corpus rather than assumed.

Three findings worth stating plainly:

- **BM25 alone beats dense embeddings on this corpus** (recall@3 0.800 vs 0.762). Pitch decks are dense with proper nouns and figures — `₹11.5 Cr`, `VizSort-M`, `$8.9B`, `Q4 2024` — precisely where lexical matching wins and a 384-dim sentence embedding blurs. Fusing the two beats either alone.
- **The ranking holds as the corpus grows.** The same 72 questions were scored at 5 decks / 212 chunks, then 8 / 286, then 18 / 730 — a 3.4× larger haystack. Absolute recall drifts down as it must, but the ordering of the five strategies never changes and the gap between the best and the dense baseline widens (+22.8% → +23.2%). A ranking that survives a tripling of the corpus is worth more than a higher number on a small one.
- **Re-ranking is the dominant cost.** It buys +8 points of recall@3 over `hybrid` and roughly +0.10 MRR, for well over an order of magnitude more latency, and it needs ~250 MB of RAM the free-tier deployment does not have. So the deployed service runs `hybrid` and the re-ranked modes are measured but not served — a real decision the table made possible rather than a guess.

Latency is reported separately in section 3; it is machine-dependent and the two harnesses disagree on the cheap modes, so it should not be read as precisely as the ranking metrics.

`python scripts/measure_retrieval.py` — offline.

### 3. Index persistence and latency

| chunks | cold build | warm load | speedup |
| ---: | ---: | ---: | :--- |
| 730 | 14.09 s | 0.029 s | **494×** |

Query latency, p50 / p95 over the 72 questions: `bm25` 7 / 16 ms, `dense`
13 / 49 ms, `hybrid` 199 / 271 ms, `hybrid+rerank` 3611 / 7634 ms. The
re-ranked figures are why the deployed instance serves `hybrid`.

The original implementation rebuilt the FAISS index on every process start and never persisted it. The index is now fingerprinted over the embedding-model id plus every chunk, so it rebuilds when the corpus or the model actually changes and loads from disk otherwise.

`python scripts/measure_latency.py` — offline.

### 4. Answer quality — preliminary

| system | correct | faithfulness | hallucination | mean latency |
| :--- | ---: | ---: | ---: | ---: |
| baseline (one-shot RAG) | 90.9% | 100% | 0% | 2.9 s |
| corrective agent | 100%* | 100% | 0% | 3.6 s |

> **Not yet resume-grade.** n=12, and the free-tier quota exhausted mid-run: the baseline completed 11 of 12 questions and the corrective agent only 7 (\*hence its 100%). The judge also fell back from the `gemini-3.1-pro` tier into flash, so the "stronger, different judge family" property did not hold for this run. Re-run with quota available:
> `python scripts/measure_answers.py`

What the corrective loop actually buys, on a query the one-shot arm fails — bare acronyms are the weak spot, because neither BM25 nor a sentence embedding connects `TAM` to the words on the slide:

| | baseline | corrective agent |
| :--- | :--- | :--- |
| question | *What is SensoVision TAM?* | *(same)* |
| retrieved slides | 1, 17, 9, 4 | **3** |
| rounds | 1 | 2 |
| rewritten query | — | `SensoVision TAM Total Addressable Market market size market opportunity` |
| answer | "I cannot find the answer in the provided context." | **"$8.7B for the Global AI Vision Market" (slide 3)** |

The baseline abstains rather than inventing a number, which is the intended behaviour; the agent grades its own context as insufficient, expands the acronym, and retrieves again.

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
    embedder.py        Cached ONNX MiniLM singleton.
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

**Embedding and re-ranking run on ONNX Runtime, not PyTorch.** Same weights, outputs identical to float32 precision, ~298 MB resident instead of ~523 MB. That is what lets the whole service run on a free 512 MB container.

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

### Deploying on a free tier

Embedding and re-ranking run on **ONNX Runtime rather than PyTorch**. Same MiniLM
weights, outputs identical to float32 precision (verified cosine `1.000000`
against sentence-transformers, and every metric in the ablation above is
unchanged) — but the resident set is a third the size:

| stack | resident | fits Render free (512 MB)? |
| :--- | ---: | :--- |
| torch + sentence-transformers | 523 MB | no — OOM |
| **ONNX Runtime** | **298 MB** | yes, 214 MB headroom |

Torch alone cost 182 MB on import and another 272 MB once MiniLM loaded, for
models that are ~90 MB each. Dropping it also removes roughly 2 GB from the
install and shrinks the image.

`render.yaml` is a Blueprint: point Render at the repo, set `GEMINI_API_KEY` in
the dashboard, deploy. `plan: free` — no card required. Free instances sleep
after 15 minutes idle and take ~1 minute to wake, which is fine for a demo.

The trade: ONNX re-ranking is slower than torch on this hardware (p50 901 ms →
2023 ms for `hybrid+rerank`). Memory was the binding constraint, not latency, so
that is the right side of the trade here — and `hybrid` without re-ranking is
14 ms at recall@3 0.892 if latency matters more than the last 6 points.

`scripts/ingest_corpus.py --text-only` disables vision entirely — that is the baseline extractor from the coverage table, runnable as a real pipeline rather than a hypothetical.

---

## Method notes

- **Ground truth is human-curated.** The 72 questions in `data/eval/eval_set.json` were written by reading the ingested transcripts, with gold slide numbers verified against the corpus. `scripts/build_eval_set.py` generates candidates with an LLM and drops any whose gold slides do not exist, but the shipped set is hand-checked.
- **The judge is a different, stronger model family than the generator** (`gemini-3.1-pro` vs `gemini-3-flash`) so the system does not grade its own output. When the pro tier is quota-exhausted the client falls back into flash and the run's LLM-usage table records it.
- **Deduping before scoring.** A slide is indexed as both a transcript chunk and a summary chunk, so a retriever legitimately returns it twice; identifiers are deduped preserving rank order before any metric reads a position, or precision and nDCG would both be inflated.
- **Corpus.** 18 of 19 decks are ingested (730 chunks); section 1 covers all 19 because it is a pure property of the PDFs and needs no ingestion. `scripts/ingest_corpus.py` is resumable and skips decks already present.
- **Free-tier quota was mostly a self-inflicted wound.** The API returns the same 429 for a per-day limit and a per-minute one. Reading every 429 as terminal made the client discard each model in turn and fail a whole deck in seconds, over a limit that clears in under one. Only per-day exhaustion is terminal now; everything else honours the API's own retryDelay. Decks that had failed four times ingest in ~45 s. The chain also leads with the flash-lite family, which carries a far larger daily allowance than the 20/day full flash models.
- **A deck that extracts nothing is never saved.** Per-page degradation means one unreadable page does not cost the other fifty, but a deck where *every* page failed is a different event — it raises rather than persisting an empty document, because both `ingest_all` and `--skip-existing` decide what to re-ingest by checking whether the output file exists, and an empty save would make that deck permanently unretryable.

## Corpus

18 of 19 decks are ingested (`data/documents/`, committed so the ablation reproduces offline). 18 public decks from the [`skyforclouds/pitch-deckz`](https://huggingface.co/datasets/skyforclouds/pitch-deckz) dataset (Airbnb, Uber, Dropbox, Facebook, LinkedIn, Coinbase, Square, Shopify, WeWork, Monzo, Revolut, Brex, Front, Mixpanel, Transferwise, Oscar Health, Dwolla, Nium) plus one private deck.
