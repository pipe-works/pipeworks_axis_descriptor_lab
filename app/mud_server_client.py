"""
app/mud_server_client.py
-----------------------------------------------------------------------------
Synchronous HTTP client for the mud server's lab API endpoints.

This module provides a thin wrapper around the mud server's REST API,
enabling the Axis Descriptor Lab to delegate OOC→IC translation to the
mud server's canonical pipeline instead of running its own Ollama calls.

Three translation modes are supported (selected by environment variable):

- **Server (prod)** — ``MUD_SERVER_URL=https://api.pipe-works.org``
- **Server (local)** — ``MUD_SERVER_URL=http://localhost:8000``
- **Standalone** — ``MUD_SERVER_URL`` unset (lab's own Ollama pipeline)

The client stores the session token in memory only — never written to disk.
On any 401 response from a lab endpoint the cached token is cleared and
:class:`MudServerSessionExpiredError` is raised so the caller can signal
the frontend to re-authenticate.

Connection management
---------------------
A persistent ``httpx.Client`` is created once in ``__init__`` and reused for
all subsequent requests.  This ensures TCP connections and TLS sessions are
pooled across calls, avoiding the cost of a fresh TLS handshake on every
request to a remote HTTPS server.  ``httpx.Client`` is thread-safe, which
is required since FastAPI runs sync handlers in a thread-pool executor.

Sync rationale
--------------
The lab's route handlers are synchronous (FastAPI runs them in a
thread-pool executor), so a blocking httpx call here matches the
existing pattern used by :class:`~app.chat_renderer.ChatRenderer`.
"""

from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_MUD_SERVER_URL: str | None = os.getenv("MUD_SERVER_URL")
if _MUD_SERVER_URL:  # pragma: no cover — env-dependent import-time init
    _MUD_SERVER_URL = _MUD_SERVER_URL.rstrip("/")

_MUD_SERVER_TIMEOUT: float = float(os.getenv("MUD_SERVER_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MudServerSessionExpiredError(Exception):
    """Raised when the mud server returns 401 (session invalid/expired)."""


class MudServerConnectionError(Exception):
    """Raised when the mud server is unreachable or a request times out."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MudServerClient:
    """Synchronous client for the mud server's lab API endpoints.

    Maintains an in-memory session token obtained via :meth:`login`.
    All lab endpoint calls attach the session_id in the request body
    (POST) or query params (GET), matching the mud server's pattern.

    A persistent ``httpx.Client`` is created once and reused for all
    requests, enabling TCP/TLS connection pooling to the remote server.

    Args:
        base_url: Mud server base URL (e.g. ``'https://api.pipe-works.org'``).
        timeout:  HTTP read timeout in seconds.  Defaults to 120.
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout, connect=10.0)
        self._client = httpx.Client(timeout=self._timeout)
        self._session_id: str | None = None
        self._role: str | None = None
        self._selected_world_id: str | None = None

    def close(self) -> None:
        """Close the underlying httpx.Client and release connections."""
        self._client.close()

    # -- Properties --------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        """True when a session token is cached in memory."""
        return self._session_id is not None

    @property
    def selected_world_id(self) -> str | None:
        """Currently selected world ID, or None."""
        return self._selected_world_id

    # -- Auth --------------------------------------------------------------

    def login(self, username: str, password: str) -> dict:
        """POST /login → store session_id + role in memory.

        Returns:
            The full LoginResponse dict from the mud server.

        Raises:
            httpx.HTTPStatusError: Non-2xx response (e.g. 401 bad credentials).
            MudServerConnectionError: Server unreachable or timed out.
        """
        try:
            resp = self._client.post(
                f"{self._base_url}/login",
                json={"username": username, "password": password},
                headers={"X-Client-Type": "axis-descriptor-lab"},
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise MudServerConnectionError(
                f"Cannot connect to mud server at {self._base_url}"
            ) from exc

        if data.get("success") and data.get("session_id"):
            self._session_id = data["session_id"]
            self._role = data.get("role")
            logger.info("Mud server login successful (role=%s)", self._role)
        else:
            self._session_id = None
            self._role = None

        return data

    def logout(self) -> None:
        """POST /logout → clear in-memory session.

        Always clears the local session, even if the server call fails.
        """
        if self._session_id:
            try:
                self._client.post(
                    f"{self._base_url}/logout",
                    json={"session_id": self._session_id},
                )
            except Exception:
                logger.warning("Mud server logout request failed (ignored)")
        self._session_id = None
        self._role = None
        self._selected_world_id = None

    def session_status(self) -> dict:
        """Return current auth status without contacting the server."""
        return {
            "authenticated": self.is_authenticated,
            "role": self._role,
        }

    # -- World selection ---------------------------------------------------

    def select_world(self, world_id: str) -> None:
        """Store the selected world_id in memory."""
        self._selected_world_id = world_id

    # -- Lab API proxies ---------------------------------------------------

    def list_worlds(self) -> list[dict]:
        """GET /api/lab/worlds → return worlds list.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """
        result = self._get("/api/lab/worlds")
        # The mud server wraps the list: {"worlds": [...]}.
        if isinstance(result, dict) and "worlds" in result:
            result = result["worlds"]
        if not isinstance(result, list):  # pragma: no cover
            raise TypeError(f"Expected list from /api/lab/worlds, got {type(result).__name__}")
        return result

    def world_config(self, world_id: str) -> dict:
        """GET /api/lab/world-config/{world_id} → return config.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """
        result = self._get(f"/api/lab/world-config/{world_id}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(f"Expected dict from world-config, got {type(result).__name__}")
        return result

    def world_prompts(self, world_id: str) -> dict:
        """GET /api/lab/world-prompts/{world_id} → return prompts list.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """
        result = self._get(f"/api/lab/world-prompts/{world_id}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(f"Expected dict from world-prompts, got {type(result).__name__}")
        return result

    def translate(
        self,
        *,
        world_id: str,
        axes: dict,
        channel: str,
        ooc_message: str,
        character_name: str = "Lab Subject",
        seed: int = -1,
        temperature: float = 0.7,
        prompt_template_override: str | None = None,
    ) -> dict:
        """POST /api/lab/translate → return LabTranslateResponse dict.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """
        body = {
            "session_id": self._session_id,
            "world_id": world_id,
            "axes": axes,
            "channel": channel,
            "ooc_message": ooc_message,
            "character_name": character_name,
            "seed": seed,
            "temperature": temperature,
        }
        if prompt_template_override is not None:
            body["prompt_template_override"] = prompt_template_override
        return self._post("/api/lab/translate", body)

    # -- Internal HTTP helpers ---------------------------------------------

    def _get(self, path: str) -> dict | list:
        """Perform an authenticated GET request with session_id as query param."""
        if not self._session_id:
            raise MudServerSessionExpiredError("Not authenticated")
        try:
            resp = self._client.get(
                f"{self._base_url}{path}",
                params={"session_id": self._session_id},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise MudServerConnectionError(
                f"Cannot connect to mud server at {self._base_url}"
            ) from exc

        if resp.status_code == 401:
            self._session_id = None
            self._role = None
            raise MudServerSessionExpiredError("Session expired or invalid")

        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        """Perform an authenticated POST request with session_id in body."""
        if not self._session_id:
            raise MudServerSessionExpiredError("Not authenticated")
        try:
            resp = self._client.post(
                f"{self._base_url}{path}",
                json=body,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise MudServerConnectionError(
                f"Cannot connect to mud server at {self._base_url}"
            ) from exc

        if resp.status_code == 401:
            self._session_id = None
            self._role = None
            raise MudServerSessionExpiredError("Session expired or invalid")

        if resp.status_code >= 400:
            logger.warning(
                "MudServerClient._post %s → HTTP %d: %s",
                path,
                resp.status_code,
                resp.text[:500],
            )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def get_mud_client() -> MudServerClient | None:
    """Return the module-level MudServerClient, or None in standalone mode."""
    return _mud_client


_mud_client: MudServerClient | None = None
if _MUD_SERVER_URL:  # pragma: no cover — env-dependent import-time init
    _mud_client = MudServerClient(_MUD_SERVER_URL, timeout=_MUD_SERVER_TIMEOUT)
    logger.info("Mud server client initialised: %s", _MUD_SERVER_URL)


def compute_translation_mode() -> str:
    """Determine the translation mode from MUD_SERVER_URL.

    Returns:
        ``"server-prod"`` when URL contains a non-localhost domain,
        ``"server-local"`` when URL points to localhost/127.0.0.1,
        ``"standalone"`` when MUD_SERVER_URL is unset.
    """
    if not _MUD_SERVER_URL:
        return "standalone"
    lower = _MUD_SERVER_URL.lower()
    if "localhost" in lower or "127.0.0.1" in lower:
        return "server-local"
    return "server-prod"
