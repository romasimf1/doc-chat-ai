"""A tiny JSON-backed vector store.

Chunks and their embeddings are stored in one JSON file on disk
(data/store.json). No external database is needed. Retrieval is a
plain cosine similarity search over the stored embeddings with numpy.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from app import config


def _empty_store() -> dict:
    return {"documents": [], "chunks": []}


class VectorStore:
    """Holds documents and their embedded chunks, persisted as JSON."""

    def __init__(self, path: str = config.STORE_PATH):
        self.path = path
        self.data = _empty_store()
        self.load()

    # -- persistence -------------------------------------------------
    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    # -- documents ---------------------------------------------------
    def add_document(self, filename: str, chunks: list[str],
                     embeddings: list[list[float]]) -> str:
        """Store a document and its embedded chunks. Returns the doc id."""
        doc_id = uuid.uuid4().hex[:12]
        self.data["documents"].append({
            "id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })
        for text, embedding in zip(chunks, embeddings):
            self.data["chunks"].append({
                "doc_id": doc_id,
                "filename": filename,
                "text": text,
                "embedding": embedding,
            })
        self.save()
        return doc_id

    def list_documents(self) -> list[dict]:
        return list(self.data["documents"])

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks. Returns True if found."""
        before_docs = len(self.data["documents"])
        self.data["documents"] = [d for d in self.data["documents"]
                                  if d["id"] != doc_id]
        self.data["chunks"] = [c for c in self.data["chunks"]
                               if c["doc_id"] != doc_id]
        if len(self.data["documents"]) == before_docs:
            return False
        self.save()
        return True

    # -- retrieval ---------------------------------------------------
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors (0..1 for non-negative)."""
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def search(self, query_embedding: list[float],
               top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
        """Return the top-k chunks most similar to the query embedding."""
        scored = []
        for chunk in self.data["chunks"]:
            score = self.cosine_similarity(query_embedding,
                                           chunk["embedding"])
            scored.append((score, chunk))
        # Highest score first; ties keep insertion order (stable sort).
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"filename": c["filename"], "text": c["text"], "score": s}
                for s, c in scored[:top_k]]
