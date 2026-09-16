"""Application service for vector retrieval over indexed email chunks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence
from preprocessing.embeddings.embedding_service import EmbeddingService
from storage.chroma_storage import ChromaStore
class Retriever:

    def __init__(self, embedder : EmbeddingService , vector_store: ChromaStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` chunks relevant to a non-empty query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        embeddings = self.embedder.embed_texts([query])
        if not embeddings:
            return []

        raw_results = self.vector_store.search(
            query_embedding=embeddings[0], top_k=top_k
        )
        return self.normalize_results(raw_results)

    
    @staticmethod
    def normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
        """Convert Chroma's one-query nested response into chunk dictionaries.

        This only adapts the vector store's returned data; all ChromaDB calls
        remain in ``ChromaStore``.
        """
        if not raw_results:
            return []

        # Permit a future store implementation to return normalized results
        # directly without making callers depend on Chroma's response shape.
        if isinstance(raw_results, list):
            return [dict(result) for result in raw_results]

        if not isinstance(raw_results, Mapping):
            return []

        ids = Retriever.first_query_values(raw_results.get("ids"))
        documents = Retriever.first_query_values(raw_results.get("documents"))
        metadatas = Retriever.first_query_values(raw_results.get("metadatas"))

        results: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            results.append(
                {
                    "id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
                }
            )
        return results


    @staticmethod
    def first_query_values(values: Any) -> Sequence[Any]:
        """Return the values for Chroma's first (and only) query vector."""
        if not values:
            return []
        if isinstance(values, (list, tuple)) and values:
            first = values[0]
            return first if isinstance(first, (list, tuple)) else values
        return []
