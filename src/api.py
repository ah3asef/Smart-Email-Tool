"""HTTP API for querying the indexed Smart Email Tool inbox."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from services.GenerationService import GenerationResult, GenerationService
from services.OllamaClient import OllamaConnectionError, OllamaError

logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Email Tool API", version="0.1.0")

_fetch_lock = threading.Lock()


class HistoryMessage(BaseModel):
    """A previous user or assistant turn supplied as optional context."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class QueryRequest(BaseModel):
    """A grounded inbox question or email-draft request."""

    query: str = Field(min_length=1, max_length=10_000)
    mode: Literal["question_answering", "email_drafting"] = "question_answering"
    top_k: int = Field(default=5, ge=1, le=10)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)


class FetchEmailsRequest(BaseModel):
    """Trigger ingestion of recent Gmail messages into the vector store."""

    max_emails: int = Field(default=50, ge=1, le=500)
    label: str = Field(default="INBOX", min_length=1, max_length=200)


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    """Create expensive retrieval dependencies only when the first query arrives."""
    # Keep FastAPI startup independent of the local ML runtime. This lets the
    # health route and OpenAPI docs work before embeddings are installed.
    from applications.Retriever import Retriever
    from preprocessing.embeddings.embedding_service import EmbeddingService
    from services.OllamaClient import OllamaClient
    from services.PromptBuilder import PromptBuilder
    from storage.chroma_storage import ChromaStore

    embedder = EmbeddingService()
    retriever = Retriever(embedder, ChromaStore())
    return GenerationService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_client=OllamaClient(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight process check that does not load the embedding model."""
    return {"status": "ok"}


@app.post("/query", response_model=GenerationResult)
def query_inbox(request: QueryRequest) -> GenerationResult:
    """Retrieve relevant email chunks and generate a grounded answer or draft."""
    try:
        return get_generation_service().generate(
            query=request.query.strip(),
            mode=request.mode,
            top_k=request.top_k,
            history=[message.model_dump() for message in request.history],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except OllamaError as exc:
        logger.warning("Ollama generation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation failed") from exc
    except Exception as exc:
        logger.exception("Inbox query failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Query failed") from exc


@app.post("/fetch-emails")
def fetch_new_emails(request: FetchEmailsRequest) -> dict:
    """Fetch the latest Gmail messages, then clean, chunk, embed and index them.

    On first use this opens Google's OAuth consent page in the browser
    (redirecting to http://localhost:8080/) and caches the token under
    secrets/. Messages are downloaded, cleaned, split into chunks, embedded
    and stored in ChromaDB. Chunks that are already indexed are skipped.
    """
    from preprocessing.chunks.email_chunking import EmailChunker
    from preprocessing.cleaner.EmailCleaner import EmailCleaner
    from preprocessing.embeddings.embedding_service import EmbeddingService
    from providers.GmailProvider import GmailLoader
    from storage.chroma_storage import ChromaStore

    with _fetch_lock:
        token_path = Path(__file__).resolve().parent.parent / "secrets" / "token_gmail.json"
        loader = GmailLoader(
            token_path=str(token_path),
            max_emails=request.max_emails,
            label=request.label,
        )

        try:
            with loader:
                emails = loader.fetch_emails()
        except (ConnectionError, RuntimeError) as exc:
            logger.warning("Gmail sync failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch emails from Gmail: {exc}",
            ) from exc

        cleaner = EmailCleaner()
        chunker = EmailChunker()
        all_chunks: list[dict] = []
        for email in emails:
            cleaned = cleaner.clean(email)
            all_chunks.extend(chunker.chunk_email(cleaned))

        logger.info("Fetched %d emails, produced %d chunks.", len(emails), len(all_chunks))

        if not all_chunks:
            return {"fetched": len(emails), "chunks_indexed": 0, "label": request.label}

        embedder = EmbeddingService()
        embeddings = embedder.embed_texts([chunk["text"] for chunk in all_chunks])
        store = ChromaStore()
        stored_count = 0

        def _is_duplicate(exc: Exception) -> bool:
            message = str(exc).lower()
            return "duplicate" in message or "already exist" in message

        try:
            store.add(all_chunks, embeddings)
            stored_count = len(all_chunks)
        except Exception as exc:  # noqa: BLE001
            # ChromaDB rejects a batch that contains already-indexed IDs
            # (e.g. a label that overlaps with a previous fetch). Fall back
            # to per-chunk adds so only genuinely new chunks are stored.
            if not _is_duplicate(exc):
                raise
            for chunk, embedding in zip(all_chunks, embeddings):
                try:
                    store.add([chunk], [embedding])
                    stored_count += 1
                except Exception as inner:  # noqa: BLE001
                    if not _is_duplicate(inner):
                        raise
                    continue

        return {
            "fetched": len(emails),
            "chunks_indexed": stored_count,
            "label": request.label,
        }
