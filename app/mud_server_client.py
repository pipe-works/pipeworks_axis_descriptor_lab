"""
app/mud_server_client.py
-----------------------------------------------------------------------------
Synchronous HTTP client for the mud server's lab API endpoints.

This module provides a thin wrapper around the mud server's REST API,
enabling the Axis Descriptor Lab to delegate OOC→IC translation to the
mud server's canonical pipeline instead of running its own Ollama calls.

Environment values still provide startup defaults, but the active chat mode
is now runtime-configurable inside the app.  That lets the Chat Translation
page switch between offline mode, a local development mud server, and an
optional configured server without editing ``.env`` or restarting FastAPI.

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
import threading
from dataclasses import dataclass
from typing import TypedDict

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_MUD_SERVER_TIMEOUT: float = float(os.getenv("MUD_SERVER_TIMEOUT", "120"))
_ENV_MUD_SERVER_URL: str | None = os.getenv("MUD_SERVER_URL")
_DEV_MUD_SERVER_URL: str | None = os.getenv("MUD_SERVER_DEV_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MudServerSessionExpiredError(Exception):
    """Raised when the mud server returns 401 (session invalid/expired)."""


class MudServerConnectionError(Exception):
    """Raised when the mud server is unreachable or a request times out."""


@dataclass(frozen=True)
class MudRuntimeModeOption:
    """One runtime-selectable chat translation mode.

    Attributes:
        key: Stable frontend/backend selector key.
        label: Human-readable label for the mode selector.
        translation_mode: External mode string used by existing UI badges.
        server_url: Mud server base URL for server-backed modes, or None for
            standalone/offline mode.
    """

    key: str
    label: str
    translation_mode: str
    server_url: str | None


class MudModeOptionDict(TypedDict):
    """Serialised runtime mode option returned to the route layer."""

    key: str
    label: str
    translation_mode: str
    server_url: str | None


class MudModeConfigDict(TypedDict):
    """Serialised runtime mode configuration returned to the route layer."""

    mode_key: str
    translation_mode: str
    active_server_url: str | None
    available_modes: list[MudModeOptionDict]


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
            "selected_world_id": self._selected_world_id,
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

    def world_image_policy_bundle(self, world_id: str) -> dict:
        """GET /api/lab/world-image-policy-bundle/{world_id} → image policy bundle.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """

        result = self._get(f"/api/lab/world-image-policy-bundle/{world_id}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(
                f"Expected dict from world-image-policy-bundle, got {type(result).__name__}"
            )
        return result

    def compile_image_prompt(
        self,
        *,
        world_id: str,
        species: str,
        gender: str,
        axes: dict,
        world_context: list[str] | None = None,
        occupation_signals: list[str] | None = None,
        model_id: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """POST /api/lab/compile-image-prompt → canonical compiled prompt package.

        The mud server remains authoritative for policy/selection/provenance.
        This helper only forwards validated request fields and returns the
        server response verbatim.

        Raises:
            MudServerSessionExpiredError: Session invalid/expired.
            MudServerConnectionError: Server unreachable or timed out.
        """

        body = {
            "session_id": self._session_id,
            "world_id": world_id,
            "species": species,
            "gender": gender,
            "axes": axes,
            "world_context": list(world_context or []),
            "occupation_signals": list(occupation_signals or []),
            "model_id": model_id,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
        }
        return self._post("/api/lab/compile-image-prompt", body)

    def create_world_prompt_draft(
        self,
        *,
        world_id: str,
        draft_name: str,
        content: str,
        based_on_name: str | None = None,
    ) -> dict:
        """POST /api/lab/world-prompts/{world_id}/drafts → create one server draft."""

        body = {
            "session_id": self._session_id,
            "draft_name": draft_name,
            "content": content,
            "based_on_name": based_on_name,
        }
        return self._post(f"/api/lab/world-prompts/{world_id}/drafts", body)

    def world_prompt_drafts(self, world_id: str) -> dict:
        """GET /api/lab/world-prompts/{world_id}/drafts → list server prompt drafts."""

        result = self._get(f"/api/lab/world-prompts/{world_id}/drafts")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(f"Expected dict from world-prompt drafts, got {type(result).__name__}")
        return result

    def world_prompt_draft(self, world_id: str, draft_name: str) -> dict:
        """GET /api/lab/world-prompts/{world_id}/drafts/{name} → load one server draft."""

        result = self._get(f"/api/lab/world-prompts/{world_id}/drafts/{draft_name}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(f"Expected dict from world-prompt draft, got {type(result).__name__}")
        return result

    def promote_world_prompt_draft(
        self,
        *,
        world_id: str,
        draft_name: str,
        target_name: str,
    ) -> dict:
        """POST /api/lab/world-prompts/{world_id}/drafts/{name}/promote → promote one draft."""

        body = {
            "session_id": self._session_id,
            "target_name": target_name,
        }
        return self._post(f"/api/lab/world-prompts/{world_id}/drafts/{draft_name}/promote", body)

    def world_policy_bundle(self, world_id: str) -> dict:
        """GET /api/lab/world-policy-bundle/{world_id} → return normalized policy bundle."""

        result = self._get(f"/api/lab/world-policy-bundle/{world_id}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(f"Expected dict from world-policy-bundle, got {type(result).__name__}")
        return result

    def create_world_policy_bundle_draft(
        self,
        *,
        world_id: str,
        draft_name: str,
        content: dict,
        based_on_name: str | None = None,
    ) -> dict:
        """POST /api/lab/world-policy-bundle/{world_id}/drafts → create one server draft."""

        body = {
            "session_id": self._session_id,
            "draft_name": draft_name,
            "content": content,
            "based_on_name": based_on_name,
        }
        return self._post(f"/api/lab/world-policy-bundle/{world_id}/drafts", body)

    def world_policy_bundle_drafts(self, world_id: str) -> dict:
        """GET /api/lab/world-policy-bundle/{world_id}/drafts → list server draft bundles."""

        result = self._get(f"/api/lab/world-policy-bundle/{world_id}/drafts")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(
                f"Expected dict from world-policy-bundle drafts, got {type(result).__name__}"
            )
        return result

    def world_policy_bundle_draft(self, world_id: str, draft_name: str) -> dict:
        """GET /api/lab/world-policy-bundle/{world_id}/drafts/{name} → load one server draft."""

        result = self._get(f"/api/lab/world-policy-bundle/{world_id}/drafts/{draft_name}")
        if not isinstance(result, dict):  # pragma: no cover
            raise TypeError(
                f"Expected dict from world-policy-bundle draft, got {type(result).__name__}"
            )
        return result

    def promote_world_policy_bundle_draft(
        self,
        *,
        world_id: str,
        draft_name: str,
    ) -> dict:
        """POST /api/lab/world-policy-bundle/{world_id}/drafts/{name}/promote → promote one draft."""

        body = {
            "session_id": self._session_id,
        }
        return self._post(
            f"/api/lab/world-policy-bundle/{world_id}/drafts/{draft_name}/promote", body
        )

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
# Runtime mode state
# ---------------------------------------------------------------------------


def _normalise_server_url(url: str | None) -> str | None:
    """Return a trimmed mud-server base URL or ``None`` when empty."""
    if url is None:
        return None
    stripped = url.strip()
    return stripped.rstrip("/") if stripped else None


def _classify_translation_mode(server_url: str | None) -> str:
    """Classify a server URL into the public translation-mode strings."""
    if not server_url:
        return "standalone"
    lower = server_url.lower()
    if "localhost" in lower or "127.0.0.1" in lower:
        return "server-local"
    return "server-prod"


_ENV_MUD_SERVER_URL = _normalise_server_url(_ENV_MUD_SERVER_URL)
_DEV_MUD_SERVER_URL = _normalise_server_url(_DEV_MUD_SERVER_URL)
_RUNTIME_DEV_SERVER_URL: str | None = _DEV_MUD_SERVER_URL


def _build_mode_options() -> tuple[MudRuntimeModeOption, ...]:
    """Build the set of runtime-selectable modes from env defaults."""
    options = [
        MudRuntimeModeOption(
            key="standalone",
            label="Offline",
            translation_mode="standalone",
            server_url=None,
        )
    ]

    if _RUNTIME_DEV_SERVER_URL:
        options.append(
            MudRuntimeModeOption(
                key="development",
                label="Development server",
                translation_mode=_classify_translation_mode(_RUNTIME_DEV_SERVER_URL),
                server_url=_RUNTIME_DEV_SERVER_URL,
            )
        )

    if _ENV_MUD_SERVER_URL and _ENV_MUD_SERVER_URL != _RUNTIME_DEV_SERVER_URL:
        options.append(
            MudRuntimeModeOption(
                key="configured",
                label="Production Server",
                translation_mode=_classify_translation_mode(_ENV_MUD_SERVER_URL),
                server_url=_ENV_MUD_SERVER_URL,
            )
        )

    return tuple(options)


def _default_mode_key() -> str:
    """Choose the startup mode key from environment defaults.

    Startup defaults to standalone/local mode so the app does not immediately
    prompt for mud-server authentication on first load.
    """
    return "standalone"


_RUNTIME_LOCK = threading.RLock()
_MUD_CLIENTS: dict[str, MudServerClient] = {}
_ACTIVE_MODE_KEY: str = _default_mode_key()


def _mode_option_map() -> dict[str, MudRuntimeModeOption]:
    """Return the current runtime mode options keyed by selector key."""
    return {option.key: option for option in _build_mode_options()}


def _resolve_active_mode_key() -> str:
    """Return the active mode key, falling back when it is no longer valid."""
    if _ACTIVE_MODE_KEY in _mode_option_map():
        return _ACTIVE_MODE_KEY
    return _default_mode_key()


def list_mud_mode_options() -> list[MudModeOptionDict]:
    """Return all runtime-selectable mud translation modes.

    The response is serialisable so the route layer can expose it directly to
    the frontend for mode-selector construction.
    """
    return [
        {
            "key": option.key,
            "label": option.label,
            "translation_mode": option.translation_mode,
            "server_url": option.server_url,
        }
        for option in _build_mode_options()
    ]


def get_mud_mode_config() -> MudModeConfigDict:
    """Return the current runtime mud-mode configuration."""
    with _RUNTIME_LOCK:
        option = _mode_option_map()[_resolve_active_mode_key()]
        return {
            "mode_key": option.key,
            "translation_mode": option.translation_mode,
            "active_server_url": option.server_url,
            "available_modes": list_mud_mode_options(),
        }


def set_mud_mode(mode_key: str, server_url: str | None = None) -> MudModeConfigDict:
    """Activate a runtime mud mode by selector key.

    Args:
        mode_key: One of the keys returned by :func:`list_mud_mode_options`.
        server_url: Optional mud-server URL override for development mode.

    Raises:
        ValueError: The requested mode key is not available in this process.
    """
    global _ACTIVE_MODE_KEY, _RUNTIME_DEV_SERVER_URL

    with _RUNTIME_LOCK:
        if mode_key == "development" and server_url is not None:
            normalised_server_url = _normalise_server_url(server_url)
            if not normalised_server_url:
                raise ValueError("Development server URL cannot be empty.")
            _RUNTIME_DEV_SERVER_URL = normalised_server_url

        if mode_key not in _mode_option_map():
            raise ValueError(f"Unknown mud mode '{mode_key}'.")
        _ACTIVE_MODE_KEY = mode_key
        option = _mode_option_map()[_resolve_active_mode_key()]
        logger.info(
            "Mud runtime mode set: key=%s translation_mode=%s url=%s",
            option.key,
            option.translation_mode,
            option.server_url,
        )
        return {
            "mode_key": option.key,
            "translation_mode": option.translation_mode,
            "active_server_url": option.server_url,
            "available_modes": list_mud_mode_options(),
        }


def get_mud_client() -> MudServerClient | None:
    """Return the active mud client for the current runtime mode, if any."""
    with _RUNTIME_LOCK:
        option = _mode_option_map()[_resolve_active_mode_key()]
        if option.server_url is None:
            return None

        client = _MUD_CLIENTS.get(option.server_url)
        if client is None:
            client = MudServerClient(option.server_url, timeout=_MUD_SERVER_TIMEOUT)
            _MUD_CLIENTS[option.server_url] = client
            logger.info("Mud server client initialised: %s", option.server_url)
        return client


def close_all_mud_clients() -> None:
    """Close every cached mud client and clear the runtime client cache."""
    with _RUNTIME_LOCK:
        for client in _MUD_CLIENTS.values():
            client.close()
        _MUD_CLIENTS.clear()


def compute_translation_mode() -> str:
    """Return the current public translation-mode string for the active mode."""
    with _RUNTIME_LOCK:
        return _mode_option_map()[_resolve_active_mode_key()].translation_mode
