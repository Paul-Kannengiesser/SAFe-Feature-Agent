"""Retrieval (building block 2): structured via `IssueFilter`, semantic via Qwen3-Embedding and Chroma.

The index is rebuilt only when the backlog has changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import chromadb
import ollama
from chromadb.config import Settings as ChromaSettings

from .backlog import BacklogItem
from .config import settings
from .sources import BacklogSource, IssueFilter

COLLECTION = "backlog"
QUERY_INSTRUCTION = (
    "Given a description of a planned software feature, "
    "retrieve existing backlog items (features and epics) that are similar or related"
)
EMBED_BATCH = 32


@dataclass
class Hit:
    item: BacklogItem
    similarity: float  # cosine similarity, 1.0 = identical


class Retriever:
    def __init__(self, source: BacklogSource, client: Any | None = None):
        self.source = source
        self.items = {item.key: item for item in source.all_items()}
        self.client = client or chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._ensure_index()

    def structured(self, query: IssueFilter) -> list[BacklogItem]:
        return self.source.search(query)

    def get(self, key: str) -> BacklogItem | None:
        return self.items.get(key)

    def epics(self) -> list[BacklogItem]:
        return [i for i in self.items.values() if i.issue_type == "Epic"]

    def similar(
        self,
        text: str,
        k: int = 5,
        issue_type: str | None = "Feature",
        exclude_epic: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[Hit]:
        where: list[dict] = []
        if issue_type:
            where.append({"issue_type": issue_type})
        if exclude_epic:
            where.append({"epic_key": {"$ne": exclude_epic}})
        result = self.collection.query(
            query_embeddings=[embed_query(text)],
            n_results=k,
            where=where[0] if len(where) == 1 else ({"$and": where} if where else None),
        )
        hits = []
        for key, distance in zip(result["ids"][0], result["distances"][0]):
            similarity = 1.0 - distance
            if key in self.items and similarity >= min_similarity:
                hits.append(Hit(self.items[key], similarity))
        return hits

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(settings.embedding_model.encode())
        for item in sorted(self.items.values(), key=lambda i: i.key):
            digest.update(f"{item.key}|{item.updated}|{len(item.description)}".encode())
        return digest.hexdigest()[:16]

    def _ensure_index(self):
        fingerprint = self._fingerprint()
        try:
            existing = self.client.get_collection(COLLECTION, embedding_function=None)
            if (existing.metadata or {}).get("fingerprint") == fingerprint:
                return existing
            self.client.delete_collection(COLLECTION)
        except Exception:  # collection does not exist yet
            pass
        collection = self.client.create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"fingerprint": fingerprint},
            embedding_function=None,
        )
        items = list(self.items.values())
        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start : start + EMBED_BATCH]
            documents = [item.context_text() for item in batch]
            collection.add(
                ids=[item.key for item in batch],
                documents=documents,
                embeddings=embed_documents(documents),
                metadatas=[_metadata(item) for item in batch],
            )
        return collection


def _metadata(item: BacklogItem) -> dict[str, Any]:
    return {"issue_type": item.issue_type, "status": item.status,
            "epic_key": item.epic_key or "", "pi": item.pi if item.pi is not None else -1}


def embed_documents(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=settings.embedding_model, input=texts).embeddings


def embed_query(text: str) -> list[float]:
    # Qwen3-Embedding expects an instruction prefix for queries, not for documents.
    prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
    return ollama.embed(model=settings.embedding_model, input=[prompt]).embeddings[0]
