"""
email_loader.py

Abstract base class defining the contract every email loader in the RAG
ingestion pipeline must follow.

Single Responsibility Principle
--------------------------------
An EmailLoader is responsible ONLY for:
    1. Connecting to an email provider / service.
    2. Fetching raw emails and converting them into `Email` model objects.
    3. Disconnecting and releasing any held resources.

It must NEVER perform:
    - Text cleaning / normalization
    - Chunking
    - Embedding generation
    - Vector database storage
    - Retrieval / querying

Data flow contract:
    Email Server  --(EmailLoader)-->  List[Email]

Downstream RAG pipeline stages (cleaning, chunking, embedding, storage,
retrieval) consume the `List[Email]` this loader returns; they are
implemented elsewhere and are out of scope for this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from models.EmailModel import Email


class EmailLoaderError(Exception):
    """Base exception for all email loader errors."""


class EmailLoaderConnectionError(EmailLoaderError):
    """Raised when a loader fails to establish or maintain a connection."""


class EmailLoaderFetchError(EmailLoaderError):
    """Raised when a loader fails to fetch or parse one or more emails."""


class EmailLoader(ABC):
    """
    Abstract base class for email loaders.

    Concrete subclasses (e.g. GmailLoader, OutlookLoader, ImapLoader) must
    implement `connect`, `fetch_emails`, and `disconnect`. The class also
    supports the context-manager protocol so it can be used as:

        with GmailLoader(...) as loader:
            emails = loader.fetch_emails()
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish a connection/session with the email provider.

        Implementations should raise `EmailLoaderConnectionError` (or a
        subclass) on failure rather than letting provider-specific
        exceptions leak out.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_emails(self) -> List[Email]:
        """
        Fetch emails from the provider and return them as a list of
        validated `Email` model objects.

        Must only be called after a successful `connect()`.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection/session and release any resources."""
        raise NotImplementedError

    def __enter__(self) -> "EmailLoader":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()