"""
services/auth_service.py

AuthService — the Google OAuth 2.0 domain logic. Everything that is not
HTTP handling lives here:

- Build the Google authorization URL (with CSRF state).
- Validate the state echoed back on the callback.
- Exchange the authorization code for tokens.
- Fetch the Google user profile.
- Create or update the user row in PostgreSQL (via database/queries).
- Refresh expired access tokens.

The route layer stays thin and only translates between HTTP and this service.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from helpers.config import Settings, get_settings
from queries import UserQueries
from models.schemas import UserModel

logger = logging.getLogger(__name__)

# How long a generated OAuth `state` remains valid before it is rejected.
STATE_TTL_SECONDS = 600  # 10 minutes

# Simple in-process CSRF state store. Replace with Redis / signed tokens if
# the app is ever scaled horizontally.
_STATE_LOCK = threading.Lock()
_STATE_STORE: dict[str, float] = {}


class AuthError(Exception):
    """Base class for all auth-flow failures."""


class StateValidationError(AuthError):
    """The OAuth `state` parameter was missing, stale, or unknown."""


class OAuthExchangeError(AuthError):
    """Code->token exchange or profile fetch failed."""


class TokenRefreshError(AuthError):
    """Could not refresh an expired access token."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuthService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.oauth_flows = {}

    # ------------------------------------------------------------------
    # OAuth URL
    # ------------------------------------------------------------------

    def get_authorization_url(self) -> tuple[str, str]:
        """
        Build the Google authorization page URL.

        Returns (authorization_url, state). The caller is expected to store
        `state` server-side so the callback can be validated (CSRF defence).
        """
        flow = Flow.from_client_config(
            self.settings.google_client_config, scopes=self.settings.google_scopes
        )
        flow.redirect_uri = self.settings.GOOGLE_REDIRECT_URI

        authorization_url, state = flow.authorization_url(
            access_type="offline",          # request a refresh_token
            prompt="consent",               # force consent so refresh_token is re-issued
            include_granted_scopes="true",
        )
        # save the same flow
        self.oauth_flows[state] = flow
        logger.info("Stored states: %s",list(self.oauth_flows.keys()))
        logger.info("Generated OAuth authorization URL with state=%s...", state[:8])
        return authorization_url, state

    # ------------------------------------------------------------------
    # Callback / authentication
    # ------------------------------------------------------------------

    def authenticate(self, code: str, state: str) -> UserModel:
        """
        Full callback handling: validate state, exchange the code, fetch the
        Google profile, and persist/refresh the user. Returns the stored user.
        """
        flow = self.oauth_flows.pop(state, None)

        if flow is None:
            raise StateValidationError("Flow not found")

        logger.info("Flow found")
        logger.info("Redirect URI: %s", flow.redirect_uri)

        logger.info("Starting token exchange")

        try:
            flow.oauth2session.scope = None
            flow.fetch_token(code=code)

        except Exception as exc:
            logger.exception("Token exchange failed")
            raise OAuthExchangeError(
                f"Failed to exchange authorization code: {exc}"
            ) from exc

        credentials = flow.credentials
        profile = self._get_profile(credentials)
        return self._save_user(profile, credentials)

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def refresh_access_token(self, refresh_token: str) -> tuple[str, datetime]:
        """Exchange a stored refresh_token for a fresh access token."""
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.settings.client_id,
            client_secret=self.settings.client_secret,
        )
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            logger.error("Token refresh failed: %s", exc)
            raise TokenRefreshError(f"Failed to refresh access token: {exc}") from exc
        if not creds.token:
            raise TokenRefreshError("Token refresh returned no access token.")
        return creds.token, creds.expiry

    # ------------------------------------------------------------------
    # Google profile + persistence
    # ------------------------------------------------------------------

    def _get_profile(self, credentials: Credentials) -> dict[str, Any]:
        """Fetch identity info (id, email, name, picture) for the token holder."""
        try:
            service = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
            profile = service.userinfo().get().execute()
        except HttpError as exc:
            logger.error("Profile fetch failed: %s", exc)
            raise OAuthExchangeError(f"Failed to fetch Google profile: {exc}") from exc
        return profile

    def _save_user(self, profile: dict[str, Any], credentials: Credentials) -> UserModel:
        """
        Create the user if new, otherwise refresh tokens/profile. Persists via
        the database layer and returns the resulting `User` model.
        """
        email = (profile.get("email") or "").strip().lower()
        google_id = str(profile.get("id") or "").strip()
        name = (profile.get("name") or "").strip()

        if not email or not google_id:
            raise OAuthExchangeError(
                "Google profile is missing required identity fields (email/id)."
            )

        access_token = credentials.token
        refresh_token = credentials.refresh_token
        token_expiry = credentials.expiry
        if token_expiry is not None and token_expiry.tzinfo is None:
            token_expiry = token_expiry.replace(tzinfo=timezone.utc)

        existing = UserQueries.find_user_by_email(email)

        if existing is not None:
            UserQueries.update_user_tokens(
                existing["id"],
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=token_expiry,
            )
            UserQueries.update_user_profile(existing["id"], name=name)
            row = UserQueries.find_user_by_email(email)
        else:
            row = UserQueries.create_user(
                google_id=google_id,
                name=name,
                email=email,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=token_expiry,
            )

        return UserModel(**dict(row))

    # ------------------------------------------------------------------
    # CSRF state handling
    # ------------------------------------------------------------------

    def _store_state(self, state: str) -> None:
        self._prune_states()
        with _STATE_LOCK:
            _STATE_STORE[state] = time.time()

    def _validate_state(self, state: str) -> None:
        if not state:
            raise StateValidationError("Missing OAuth state parameter.")
        with _STATE_LOCK:
            created_at = _STATE_STORE.pop(state, None)
        if created_at is None:
            raise StateValidationError("Invalid or unknown OAuth state.")
        if time.time() - created_at > STATE_TTL_SECONDS:
            raise StateValidationError("OAuth state has expired.")

    def _prune_states(self) -> None:
        cutoff = time.time() - STATE_TTL_SECONDS
        with _STATE_LOCK:
            expired = [s for s, t in _STATE_STORE.items() if t < cutoff]
            for s in expired:
                _STATE_STORE.pop(s, None)
