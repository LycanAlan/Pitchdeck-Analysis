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
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from pitchlens.agent.corrective import CorrectiveRAGAgent
from pitchlens.config import settings
from pitchlens.domain import Answer, Chunk, DeckDocument
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
# never drift from the strategies that actually exist.
RETRIEVAL_MODES = RetrieverFactory.MODES

DEFAULT_MODE = "hybrid+rerank"


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
        self.llm = GeminiClient()
        self.embedder = get_embedder()
        self.ingestion = IngestionPipeline(self.llm)
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

    def ingest(self, pdf: Path) -> DeckDocument:
        document = self.ingestion.ingest(pdf)
        document.save(settings.paths.documents / f"{document.name}.json")
        self.reload(rebuild=True)
        return document


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: str = DEFAULT_MODE
    agentic: bool = True


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


Service = Annotated[RagService, Depends(get_service)]


@app.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=service.ready,
        decks=len(service.documents),
        chunks=len(service.chunks),
        modes=list(RETRIEVAL_MODES),
        default_mode=DEFAULT_MODE,
    )


@app.get("/decks", response_model=list[DeckInfo])
def decks(service: Service) -> list[DeckInfo]:
    return [DeckInfo.from_document(d) for d in service.documents]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: Service) -> QueryResponse:
    """Answer a question. Sync `def` so FastAPI runs the blocking LLM call in its
    worker threadpool instead of stalling the event loop."""
    if not service.ready:
        raise HTTPException(409, "No decks ingested yet. POST a PDF to /ingest first.")

    started = time.perf_counter()
    answer = service.answer(request.question, request.mode, request.agentic)
    return QueryResponse.from_answer(answer, int((time.perf_counter() - started) * 1000))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(service: Service, file: Annotated[UploadFile, File()]) -> IngestResponse:
    """Upload a PDF, run the ingestion pipeline, and rebuild the index."""
    destination = settings.paths.decks / Path(file.filename or "deck.pdf").name
    destination.write_bytes(await file.read())

    started = time.perf_counter()
    document = await run_in_threadpool(service.ingest, destination)
    return IngestResponse(
        deck=DeckInfo.from_document(document),
        total_chunks=len(service.chunks),
        seconds=round(time.perf_counter() - started, 2),
    )
