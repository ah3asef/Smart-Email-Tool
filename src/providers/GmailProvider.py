"""
gmail_loader.py

Concrete `EmailLoader` implementation that connects to the Gmail API and
returns emails as validated `Email` model objects.

Data flow:
    Gmail Server  --(GmailLoader)-->  List[Email]

This module only reads and structures emails. It performs no cleaning,
chunking, embedding, storage, or retrieval — those belong to later
stages of the RAG pipeline.

Setup
-----
1. Enable the Gmail API in a Google Cloud project and create a "Web
   application" OAuth client, then download its client secrets as
   `credentials.json`.
2. In that OAuth client's "Authorized redirect URIs", add
   `http://localhost:<redirect_port>/` (default redirect_port is 8080,
   i.e. `http://localhost:8080/`). This exact match is required — Web
   application clients (unlike Desktop app clients) reject redirect
   URIs that weren't pre-registered.
3. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
4. On first run, `connect()` opens a browser for the OAuth consent
   screen, redirects to your local server on `redirect_port`, and
   caches the resulting token at `token_path` for reuse.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from base.BaseEmailLoader import EmailLoader, EmailLoaderConnectionError, EmailLoaderFetchError
from helpers.config import get_settings
from models.EmailModel import Email

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailLoader(EmailLoader):
    """
    Loads emails from a Gmail account via the Gmail API.

    Parameters
    ----------
    token_path:
        Path where the user's OAuth token is cached/loaded from, so the
        consent flow doesn't have to run on every startup.
    max_emails:
        Maximum number of emails `fetch_emails()` will return.
    label:
        Gmail label or folder to read from (e.g. "INBOX", "SENT",
        a custom label name, or a label ID).
    scopes:
        Optional override of the OAuth scopes requested. Defaults to
        read-only Gmail access.
    redirect_port:
        Fixed local port used for the OAuth loopback redirect. Required
        because "Web application" OAuth clients (unlike "Desktop app"
        clients) only accept exact, pre-registered redirect URIs — a
        randomly chosen port will be rejected by Google. In Google Cloud
        Console, under this OAuth client's "Authorized redirect URIs",
        add:  http://localhost:<redirect_port>/
    """

    def __init__(
        self,
        token_path: str,
        max_emails: int = 50,
        label: str = "INBOX",
        scopes: Optional[List[str]] = None,
        redirect_port: int = 8080,
    ) -> None:
        self.token_path = Path(token_path)
        self.max_emails = max_emails
        self.label = label
        self.scopes = scopes or DEFAULT_SCOPES
        self.redirect_port = redirect_port
        self.client_config = self._client_config_from_settings()

        self._service: Optional[Resource] = None
        self._creds: Optional[Credentials] = None

    #Loading the Project Secrets from Helpers
    @staticmethod
    def _client_config_from_settings() -> Dict[str, Any]:
        settings = get_settings()
        return {
            "web": {
                "client_id": settings.client_id,
                "project_id": settings.project_id,
                "auth_uri": settings.auth_uri,
                "token_uri": settings.token_uri,
                "auth_provider_x509_cert_url": settings.auth_provider_x509_cert_url,
                "client_secret": settings.client_secret,
            }
        }

    @property
    def is_connected(self) -> bool:
        return self._service is not None

    # ------------------------------------------------------------------
    # EmailLoader interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate via OAuth and build the Gmail API service object."""
        logger.info("Connecting to Gmail API...")
        try:
            creds = self._load_or_refresh_credentials()
            self._creds = creds
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            logger.info("Gmail API connection established.")
        except EmailLoaderConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all failures
            raise EmailLoaderConnectionError(f"Failed to connect to Gmail: {exc}") from exc

    def disconnect(self) -> None:
        """Release the Gmail API service and cached credentials."""
        if self._service is not None:
            try:
                self._service.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error while closing Gmail service: %s", exc)
        self._service = None
        self._creds = None
        logger.info("Disconnected from Gmail API.")

    def fetch_emails(self) -> List[Email]:
        """
        Fetch up to `max_emails` emails from `label` and return them as
        validated `Email` objects. Individual messages that fail to
        download or parse are logged and skipped rather than aborting
        the whole batch.
        """
        if not self.is_connected:
            raise EmailLoaderConnectionError("Not connected. Call connect() first.")

        emails: List[Email] = []
        message_ids = self._get_message_ids()

        for message_id in message_ids:
            try:
                raw_message = self._download_message(message_id)
                email_obj = self._parse_email(message_id, raw_message)
                emails.append(email_obj)
            except Exception as exc:  # noqa: BLE001 - one bad email shouldn't break the batch
                logger.warning("Skipping message %s due to error: %s", message_id, exc)
                continue

        logger.info(
            "Fetched %d/%d emails from Gmail (label=%s).",
            len(emails),
            len(message_ids),
            self.label,
        )
        return emails

    # ------------------------------------------------------------------
    # Private helpers for extracting and validating
    # ------------------------------------------------------------------

    def _load_or_refresh_credentials(self) -> Credentials:
        """Load a cached token, refresh it, or run the OAuth flow."""
        creds: Optional[Credentials] = None

        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Gmail token...")
                creds.refresh(Request())
            else:
                logger.info(
                    "Running OAuth consent flow for Gmail (redirect port %d)...",
                    self.redirect_port,
                )
                flow = InstalledAppFlow.from_client_config(
                    self.client_config, self.scopes
                )
                # A "Web application" OAuth client requires the redirect URI to
                # exactly match one registered in Cloud Console, so the port
                # must be fixed rather than left to a random OS-assigned one.
                creds = flow.run_local_server(port=self.redirect_port)

            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json())

        return creds

    def _get_message_ids(self) -> List[str]:
        """Retrieve up to `max_emails` message IDs for `label`, paginating as needed."""
        assert self._service is not None
        message_ids: List[str] = []
        page_token: Optional[str] = None

        try:
            while len(message_ids) < self.max_emails:
                remaining = self.max_emails - len(message_ids)
                response = (
                    self._service.users()
                    .messages()
                    .list(
                        userId="me",
                        labelIds=[self.label] if self.label else None,
                        maxResults=min(remaining, 500),
                        pageToken=page_token,
                    )
                    .execute()
                )
                message_ids.extend(m["id"] for m in response.get("messages", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to list Gmail messages: {exc}") from exc

        return message_ids[: self.max_emails]

    def _download_message(self, message_id: str) -> Dict[str, Any]:
        """Download the full raw message payload for a single message ID."""
        assert self._service is not None
        try:
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to fetch message {message_id}: {exc}") from exc

    def _extract_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Flatten Gmail's header list into a lowercase-keyed dict."""
        return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract the message body, preferring text/plain over text/html."""
        plain_text = self._find_body_by_mime_type(payload, "text/plain")
        if plain_text:
            return plain_text

        html_text = self._find_body_by_mime_type(payload, "text/html")
        return html_text or ""

    def _find_body_by_mime_type(self, payload: Dict[str, Any], mime_type: str) -> Optional[str]:
        """Recursively search MIME parts for the first match of `mime_type`."""
        if payload.get("mimeType") == mime_type:
            data = payload.get("body", {}).get("data")
            if data:
                return self._decode_base64(data)

        for part in payload.get("parts", []) or []:
            result = self._find_body_by_mime_type(part, mime_type)
            if result:
                return result

        return None

    def _parse_email(self, message_id: str, raw_message: Dict[str, Any]) -> Email:
        """Convert a raw Gmail API message into a validated `Email` object."""
        payload = raw_message.get("payload", {})
        headers = self._extract_headers(payload)

        subject = headers.get("subject", "")
        _, sender_addr = parseaddr(headers.get("from", ""))
        recipients = self._parse_recipients(headers.get("to", ""))
        date = self._parse_date(headers.get("date"), raw_message.get("internalDate"))
        body = self._extract_body(payload)

        return Email(
            id=message_id,
            subject=subject,
            sender=sender_addr,
            recipients=recipients,
            body=body,
            date=date,
        )

    @staticmethod
    def _parse_recipients(to_header: str) -> List[str]:
        if not to_header:
            return []
        parsed = [parseaddr(addr)[1] for addr in to_header.split(",")]
        return [addr for addr in parsed if addr]

    @staticmethod
    def _parse_date(date_header: Optional[str], internal_date_ms: Optional[str]) -> datetime:
        if date_header:
            try:
                return parsedate_to_datetime(date_header)
            except (TypeError, ValueError):
                pass

        if internal_date_ms:
            return datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)

        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _decode_base64(data: str) -> str:
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to decode message body: %s", exc)
            return ""

