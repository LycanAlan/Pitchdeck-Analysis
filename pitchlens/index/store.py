"""Dense index with on-disk persistence.

Raw FAISS rather than LangChain's wrapper: the wrapper hides the docstore behind
its own serialisation format, which is what made the previous implementation
rebuild embeddings on every app start. Here the index and the chunks it was
built from are two plain files, so a warm start is a file read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss

from ..domain import Chunk, ChunkKind, ScoredChunk
from .embedder import Embedder

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"


def _fingerprint(chunks: list[Chunk], model: str) -> str:
    """Identity of an index: its corpus *and* the model that embedded it.

    The model belongs in the hash because swapping embedding models leaves the
    corpus untouched while making every stored vector meaningless — a staleness
    bug that would otherwise surface only as quietly worse answers.
    """
    digest = hashlib.sha256(model.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


class VectorIndex:
    """A FAISS inner-product index over normalised chunk embeddings."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder: Embedder, model: str):
        self._index = index
        self._chunks = chunks
        self._embedder = embedder
        self._model = model

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> VectorIndex:
        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(embedder.encode([c.text for c in chunks]))
        return cls(index, list(chunks), embedder, embedder.name)

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> VectorIndex:
        payload = json.loads((path / _CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                deck=c["deck"],
                slide=c["slide"],
                kind=ChunkKind(c["kind"]),
                text=c["text"],
                section=c["section"],
            )
            for c in payload["chunks"]
        ]
        index = faiss.read_index(str(path / _INDEX_FILE))
        return cls(index, chunks, embedder, payload["model"])

    @classmethod
    def load_or_build(cls, path: Path, chunks: list[Chunk], embedder: Embedder) -> VectorIndex:
        """Load the cached index, rebuilding only if the corpus or model changed.

        This is the method callers use. Embedding a corpus takes minutes, so a
        rebuild has to be earned: the fingerprint comparison is what lets the
        cache be trusted instead of bypassed.
        """
        if cls.exists(path):
            cached = cls.load(path, embedder)
            if cached.fingerprint == _fingerprint(chunks, embedder.name):
                return cached

        index = cls.build(chunks, embedder)
        index.save(path)
        return index

    @staticmethod
    def exists(path: Path) -> bool:
        return (path / _INDEX_FILE).exists() and (path / _CHUNKS_FILE).exists()

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._chunks, self._model)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        # Clamping keeps FAISS from padding the result with -1 ids on small corpora.
        wanted = min(k, len(self._chunks))
        scores, ids = self._index.search(self._embedder.encode_one(query).reshape(1, -1), wanted)
        return [ScoredChunk(self._chunks[i], float(s)) for i, s in zip(ids[0], scores[0])]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        payload = {
            "model": self._model,
            "chunks": [
                {
                    "deck": c.deck,
                    "slide": c.slide,
                    "kind": c.kind.value,
                    "text": c.text,
                    "section": c.section,
                }
                for c in self._chunks
            ],
        }
        (path / _CHUNKS_FILE).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
