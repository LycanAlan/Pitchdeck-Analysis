"""FastAPI application exposing ingestion and question answering.

The whole point of this layer is that the expensive objects — the embedding
model, the FAISS index, the cross-encoder — are constructed once per process and
reused. The previous Streamlit-only design rebuilt them on every interaction,
which dominated end-to-end latency. `RagService` owns that state; the endpoints
are thin.
"""

from __future__ import annotations

import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from pitchlens.agent.corrective import CorrectiveRAGAgent
from pitchlens.config import settings
from pitchlens.domain import Answer, Chunk, DeckDocument, ScoredChunk
from pitchlens.generation.answerer import Answerer
from pitchlens.generation.base import RAGPipeline
from pitchlens.index.embedder import get_embedder
from pitchlens.index.store import VectorIndex
from pitchlens.ingest.pipeline import IngestionPipeline
from pitchlens.llm import GeminiClient
from pitchlens.retrieval.base import Retriever
from pitchlens.retrieval.factory import RetrieverFactory

# One index over the whole corpus: questions are asked across decks, and a
# per-deck index would mean rebuilding retrievers on every deck switch.
INDEX_NAME = "corpus"

# Read off the factory rather than restated, so the UI's ablation selector can
# never drift from the strategies that actually exist. Cross-encoder modes are
# withheld where the instance cannot afford the model (see
# RetrievalConfig.rerank_enabled), so the service advertises only what it can
# actually serve.
RETRIEVAL_MODES = tuple(
    m for m in RetrieverFactory.MODES
    if settings.retrieval.rerank_enabled or not m.endswith("+rerank")
)

DEFAULT_MODE = "hybrid+rerank" if settings.retrieval.rerank_enabled else "hybrid"


def _discard(path: Path) -> None:
    """Remove a persisted index so the next load genuinely rebuilds it.

    New chunks invalidate the index, but `load_or_build` cannot know that — it
    only sees a file that exists. Deleting the artifacts is the honest signal.
    """
    if path.is_dir():
        shutil.rmtree(path)
    for sibling in path.parent.glob(f"{path.name}.*"):
        sibling.unlink()


class RagService:
    """Long-lived retrieval + generation state, built once at startup.

    Retrievers are memoised per mode because the re-ranking strategies load a
    cross-encoder; without the cache, flipping the demo's ablation selector would
    pay that cost on every question.
    """

    documents: list[DeckDocument]
    chunks: list[Chunk]
    index: VectorIndex | None
    factory: RetrieverFactory | None

    def __init__(self) -> None:
        settings.paths.ensure()
        # Without a key the service still starts and serves retrieval: search,
        # /decks and /health need no model at all. Only generation and ingestion
        # do, and those report a clear 503 rather than taking the process down at
        # boot -- which previously turned a missing dashboard variable into a
        # failed deploy.
        self.llm = GeminiClient() if settings.has_api_key else None
        self.embedder = get_embedder()
        self.ingestion = IngestionPipeline(self.llm) if self.llm else None
        self._lock = threading.Lock()
        self._retrievers: dict[str, Retriever] = {}
        self._pipelines: dict[tuple[str, bool], RAGPipeline] = {}
        self.reload()

    @property
    def index_path(self) -> Path:
        return settings.paths.indices / INDEX_NAME

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def reload(self, *, rebuild: bool = False) -> None:
        """Re-read every ingested document from disk and re-open the index."""
        documents = [
            DeckDocument.load(p) for p in sorted(settings.paths.documents.glob("*.json"))
        ]
        chunks = [c for d in documents for c in d.chunks()]
        if rebuild:
            _discard(self.index_path)

        index = VectorIndex.load_or_build(self.index_path, chunks, self.embedder) if chunks else None
        factory = RetrieverFactory(index) if chunks else None

        with self._lock:
            self.documents = documents
            self.chunks = chunks
            self.index = index
            self.factory = factory
            self._retrievers.clear()
            self._pipelines.clear()

    def _cached_retriever(self, mode: str) -> Retriever:
        """Caller must hold `_lock`."""
        if mode not in self._retrievers:
            self._retrievers[mode] = self.factory.build(mode)
        return self._retrievers[mode]

    def retriever(self, mode: str) -> Retriever:
        # The lock covers only cache construction. Holding it across retrieval
        # and generation would serialise every request behind one LLM call.
        with self._lock:
            return self._cached_retriever(mode)

    def pipeline(self, mode: str, agentic: bool) -> RAGPipeline:
        """Answerer and CorrectiveRAGAgent share one interface, so the endpoint
        picks an arm here and never branches again."""
        with self._lock:
            key = (mode, agentic)
            if key not in self._pipelines:
                retriever = self._cached_retriever(mode)
                build = CorrectiveRAGAgent if agentic else Answerer
                self._pipelines[key] = build(retriever, self.llm)
            return self._pipelines[key]

    def answer(self, question: str, mode: str, agentic: bool) -> Answer:
        return self.pipeline(mode, agentic).answer(question)

    def search(self, query: str, mode: str, k: int) -> list[ScoredChunk]:
        """Retrieval with no generation, and therefore no API key."""
        return self.retriever(mode).retrieve(query, k)

    def ingest(self, pdf: Path) -> DeckDocument:
        document = self.ingestion.ingest(pdf)
        document.save(settings.paths.documents / f"{document.name}.json")
        self.reload(rebuild=True)
        return document


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: str = DEFAULT_MODE
    agentic: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = DEFAULT_MODE
    k: int = settings.retrieval.final_k


class SearchHit(BaseModel):
    deck: str
    slide: int
    kind: str
    score: float
    text: str

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> SearchHit:
        return cls(
            deck=scored.chunk.deck,
            slide=scored.chunk.slide,
            kind=scored.chunk.kind.value,
            score=round(scored.score, 4),
            text=scored.chunk.text[:400],
        )


class CitationModel(BaseModel):
    deck: str
    slide: int
    kind: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    retrieval_rounds: int
    grounded: bool
    latency_ms: int
    rewritten_query: str = ""

    @classmethod
    def from_answer(cls, answer: Answer, latency_ms: int) -> QueryResponse:
        return cls(
            answer=answer.text,
            citations=[
                CitationModel(deck=c.deck, slide=c.slide, kind=c.kind) for c in answer.citations
            ],
            retrieval_rounds=answer.retrieval_rounds,
            grounded=answer.grounded,
            latency_ms=latency_ms,
            rewritten_query=answer.rewritten_query,
        )


class DeckInfo(BaseModel):
    name: str
    slides: int
    vision_slides: int
    chunks: int

    @classmethod
    def from_document(cls, document: DeckDocument) -> DeckInfo:
        return cls(
            name=document.name,
            slides=len(document.slides),
            vision_slides=document.vision_slides,
            chunks=len(document.chunks()),
        )


class HealthResponse(BaseModel):
    status: str
    ready: bool
    decks: int
    chunks: int
    modes: list[str]
    default_mode: str
    # Retrieval always works; generation needs a key. Reporting it here means a
    # 503 from /query is diagnosable without reading the logs.
    generation_available: bool


class IngestResponse(BaseModel):
    deck: DeckInfo
    total_chunks: int
    seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.service = await run_in_threadpool(RagService)
    yield


app = FastAPI(
    title="PitchLens",
    version="2.0",
    summary="Multimodal retrieval-augmented question answering over startup pitch decks.",
    lifespan=lifespan,
)

# Wide open on purpose: this runs on localhost or inside a compose network for a
# demo. There is no authentication or per-user isolation — out of scope.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service(request: Request) -> RagService:
    return request.app.state.service


def _check_mode(mode: str) -> None:
    if mode not in RETRIEVAL_MODES:
        raise HTTPException(
            400,
            f"Unknown or unavailable retrieval mode {mode!r}. Available: {list(RETRIEVAL_MODES)}."
            + ("" if settings.retrieval.rerank_enabled else " Cross-encoder modes are disabled on this instance."),
        )


Service = Annotated[RagService, Depends(get_service)]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare hostname to the interactive docs.

    Without this, opening the deployed URL returns FastAPI's 404 for an
    undefined route, which reads as a broken deployment rather than an API with
    no landing page. /docs is the demo surface: it lists every endpoint and can
    run /search against the live corpus in the browser.
    """
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=service.ready,
        decks=len(service.documents),
        chunks=len(service.chunks),
        modes=list(RETRIEVAL_MODES),
        default_mode=DEFAULT_MODE,
        generation_available=service.llm is not None,
    )


@app.get("/decks", response_model=list[DeckInfo])
def decks(service: Service) -> list[DeckInfo]:
    return [DeckInfo.from_document(d) for d in service.documents]


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest, service: Service) -> list[SearchHit]:
    """Retrieval only — no generation, so this works with no API key configured.

    It is also the honest demo of the ablation: switch `mode` and watch which
    slides come back.
    """
    if not service.ready:
        raise HTTPException(409, "No decks ingested yet. POST a PDF to /ingest first.")
    _check_mode(request.mode)
    return [SearchHit.from_scored(s) for s in service.search(request.query, request.mode, request.k)]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: Service) -> QueryResponse:
    """Answer a question. Sync `def` so FastAPI runs the blocking LLM call in its
    worker threadpool instead of stalling the event loop."""
    if not service.ready:
        raise HTTPException(409, "No decks ingested yet. POST a PDF to /ingest first.")
    _check_mode(request.mode)
    if service.llm is None:
        raise HTTPException(
            503,
            "GEMINI_API_KEY is not configured, so answers cannot be generated. "
            "Retrieval still works — use POST /search.",
        )

    started = time.perf_counter()
    answer = service.answer(request.question, request.mode, request.agentic)
    return QueryResponse.from_answer(answer, int((time.perf_counter() - started) * 1000))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(service: Service, file: Annotated[UploadFile, File()]) -> IngestResponse:
    """Upload a PDF, run the ingestion pipeline, and rebuild the index."""
    if service.ingestion is None:
        raise HTTPException(503, "GEMINI_API_KEY is not configured; ingestion needs a vision model.")

    destination = settings.paths.decks / Path(file.filename or "deck.pdf").name
    destination.write_bytes(await file.read())

    started = time.perf_counter()
    document = await run_in_threadpool(service.ingest, destination)
    return IngestResponse(
        deck=DeckInfo.from_document(document),
        total_chunks=len(service.chunks),
        seconds=round(time.perf_counter() - started, 2),
    )
