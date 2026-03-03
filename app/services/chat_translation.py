"""
Chat translation and chat-save orchestration helpers.

This module contains the business logic previously embedded inside
``app.main`` for ``POST /api/translate_chat`` and ``POST /api/save_chat``.
The route layer delegates here for translation-mode selection, standalone
and mud-server translation flows, and chat save-package creation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from pipeworks_ipc import (
    compute_ipc_id,
    compute_output_hash,
    compute_payload_hash,
    compute_system_prompt_hash,
)

from app.chat_renderer import OLLAMA_HOST, ChatRenderer
from app.file_loaders import load_chat_default_prompt, load_prompt
from app.mud_server_client import (
    MudServerClient,
    MudServerConnectionError,
    MudServerSessionExpiredError,
    get_mud_client,
)
from app.save_formatting import build_game_log_md, build_system_prompt_md, save_folder_name
from app.schema import (
    ChatSaveRequest,
    ChatSaveResponse,
    ChatTranslationRequest,
    ChatTranslationResponse,
    ChatTranslationResult,
)

logger = logging.getLogger(__name__)


def _build_profile_summary(
    active_axes: dict[str, dict[str, str | float]], character_name: str
) -> str:
    """Build the mud-server-style ``{{profile_summary}}`` block.

    The standalone lab flow should render prompt context using the same
    summary shape as the mud server's canonical translation service:
    a leading character-name line followed by title-cased axis rows with
    two-decimal scores.  Matching this format keeps identical prompt files
    semantically aligned across both repositories.

    Args:
        active_axes: Axis snapshot keyed by axis name.  Each value must carry
            ``label`` and ``score`` keys.
        character_name: Display name to place on the first summary line.

    Returns:
        Multi-line prompt block suitable for ``{{profile_summary}}``.
    """

    lines = [f"  Character: {character_name}"]
    for axis_name, axis_value in active_axes.items():
        label = str(axis_value["label"])
        score = float(axis_value["score"])
        display_name = axis_name.replace("_", " ").title()
        lines.append(f"  {display_name}: {label} ({score:.2f})")
    return "\n".join(lines)


def _render_system_prompt(template: str, profile: dict[str, str | float], ooc_message: str) -> str:
    """Render a standalone chat prompt using the mud-server placeholder contract.

    Placeholder replacement deliberately mirrors the mud server:
    profile-derived keys are substituted first and ``{{ooc_message}}`` is
    substituted last.  This allows world prompt files copied from the server
    to behave the same way in the lab, including templates that embed the OOC
    message inside the system prompt text.

    Args:
        template: Raw prompt template text.
        profile: Flat placeholder map containing axis, channel, and
            ``profile_summary`` keys.
        ooc_message: Raw OOC user message.

    Returns:
        Fully-rendered system prompt text.
    """

    rendered = template
    for key, value in profile.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    rendered = rendered.replace("{{ooc_message}}", ooc_message)
    return rendered


def translate_chat(req: ChatTranslationRequest, prompt_root: Path) -> ChatTranslationResponse:
    """
    Translate one or two OOC chat messages using either the mud server's
    canonical pipeline or the local standalone Ollama pipeline.
    """
    if req.ollama_host:
        return _translate_standalone(req, prompt_root)

    mud_client = get_mud_client()
    if mud_client is not None:
        if not mud_client.is_authenticated:
            raise HTTPException(status_code=401, detail="Not authenticated — please log in.")
        world_id = req.world_id or mud_client.selected_world_id
        if not world_id:
            raise HTTPException(status_code=400, detail="No world selected.")
        return _translate_via_server(req, mud_client, world_id)

    return _translate_standalone(req, prompt_root)


def _translate_via_server(
    req: ChatTranslationRequest, mud_client: MudServerClient, world_id: str
) -> ChatTranslationResponse:
    """Delegate translation to the mud server's canonical pipeline."""

    def _server_translate_one(char) -> ChatTranslationResult:
        axes_for_server = {
            name: {"label": av.label, "score": av.score} for name, av in char.axes.items()
        }

        logger.debug(
            "Server translate: world_id=%r, axes=%s, channel=%r",
            world_id,
            list(axes_for_server.keys()),
            char.channel,
        )

        try:
            data = mud_client.translate(
                world_id=world_id,
                axes=axes_for_server,
                channel=char.channel,
                ooc_message=char.ooc_message,
                character_name=char.character_name or "Lab Subject",
                seed=req.seed,
                temperature=req.temperature,
                prompt_template_override=req.system_prompt or None,
            )
        except MudServerSessionExpiredError:
            raise HTTPException(
                status_code=401,
                detail="Mud server session expired. Please log in again.",
            )
        except MudServerConnectionError as exc:
            logger.warning("Server translate connection error: %s", exc)
            return ChatTranslationResult(
                ic_text=None,
                status="fallback.api_error",
                error_detail="Cannot connect to mud server.",
            )
        except Exception as exc:
            logger.warning("Server translate unexpected error: %s: %s", type(exc).__name__, exc)
            return ChatTranslationResult(
                ic_text=None,
                status="fallback.api_error",
                error_detail=f"Server error: {type(exc).__name__}",
            )

        ic_text = data.get("ic_text")
        status = data.get("status", "fallback.api_error")
        rendered_prompt = data.get("rendered_prompt", "")
        model = data.get("model", req.model)

        log_fn = logger.debug if (ic_text and status == "success") else logger.warning
        log_fn(
            "Server translate response: status=%r, ic_text=%s, model=%r, has_prompt=%s",
            status,
            f"<{len(ic_text)} chars>" if ic_text else None,
            model,
            bool(rendered_prompt),
        )

        prompt_template = data.get("prompt_template") or rendered_prompt
        sp_hash = compute_system_prompt_hash(prompt_template) if prompt_template else None

        active: set[str] = (
            set(char.active_axes) if char.active_axes is not None else set(char.axes.keys())
        )
        input_dict = {
            "axes": {k: v.model_dump() for k, v in char.axes.items() if k in active},
            "ooc_message": char.ooc_message,
            "channel": char.channel,
        }
        input_hash = compute_payload_hash(input_dict)

        if ic_text is None or status != "success":
            detail = f"Remote translation failed — server returned status '{status}'."
            if status in ("fallback.api_error", "fallback.timeout"):
                detail += " The model may still be loading in Ollama."
            return ChatTranslationResult(
                ic_text=None,
                status=status,
                input_hash=input_hash,
                system_prompt_hash=sp_hash,
                model=model,
                error_detail=detail,
            )

        out_hash = compute_output_hash(ic_text)
        ipc_id = compute_ipc_id(
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            model=model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )

        return ChatTranslationResult(
            ic_text=ic_text,
            status="success",
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            output_hash=out_hash,
            ipc_id=ipc_id,
            model=model,
        )

    result_a = _server_translate_one(req.character_a) if req.character_a is not None else None
    result_b = _server_translate_one(req.character_b) if req.character_b is not None else None

    return ChatTranslationResponse(character_a=result_a, character_b=result_b)


def _translate_standalone(
    req: ChatTranslationRequest, prompt_root: Path
) -> ChatTranslationResponse:
    """Run translation using the lab's own Ollama pipeline."""
    from app.output_validator import OutputValidator

    ollama_base = (req.ollama_host or OLLAMA_HOST).rstrip("/")

    if req.system_prompt:
        template = req.system_prompt
    elif req.prompt_name:
        template = load_prompt(req.prompt_name, purpose="chat_translation")
    else:
        template = load_chat_default_prompt()

    validator = OutputValidator(
        strict_mode=req.strict_mode,
        max_output_chars=req.max_output_chars,
    )

    def _translate_one(char, fallback_name: str) -> ChatTranslationResult:
        active: set[str] = (
            set(char.active_axes) if char.active_axes is not None else set(char.axes.keys())
        )

        profile: dict[str, str | float] = {}
        active_axes: dict[str, dict[str, str | float]] = {}
        character_name = char.character_name or fallback_name

        for axis_name, axis_val in char.axes.items():
            if axis_name not in active:
                continue
            profile[f"{axis_name}_label"] = axis_val.label
            profile[f"{axis_name}_score"] = round(axis_val.score, 3)
            active_axes[axis_name] = {
                "label": axis_val.label,
                "score": round(axis_val.score, 3),
            }

        profile["channel"] = char.channel
        profile["character_name"] = character_name
        profile["profile_summary"] = _build_profile_summary(active_axes, character_name)
        rendered = _render_system_prompt(template, profile, char.ooc_message)

        input_dict = {
            "axes": {k: v.model_dump() for k, v in char.axes.items() if k in active},
            "ooc_message": char.ooc_message,
            "channel": char.channel,
        }
        input_hash = compute_payload_hash(input_dict)
        sp_hash = compute_system_prompt_hash(template)

        renderer = ChatRenderer(
            host=ollama_base,
            model=req.model,
            timeout_seconds=120.0,
            temperature=req.temperature,
            seed=req.seed,
            max_tokens=req.max_tokens,
        )
        ic_raw = renderer.render(rendered, char.ooc_message)

        if ic_raw is None:
            return ChatTranslationResult(
                ic_text=None,
                status="fallback.api_error",
                input_hash=input_hash,
                system_prompt_hash=sp_hash,
                model=req.model,
            )

        ic_text = validator.validate(ic_raw)
        if ic_text is None:
            return ChatTranslationResult(
                ic_text=None,
                status="fallback.validation_failed",
                input_hash=input_hash,
                system_prompt_hash=sp_hash,
                model=req.model,
            )

        out_hash = compute_output_hash(ic_text)
        ipc_id = compute_ipc_id(
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.seed,
        )

        return ChatTranslationResult(
            ic_text=ic_text,
            status="success",
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            output_hash=out_hash,
            ipc_id=ipc_id,
            model=req.model,
        )

    result_a = (
        _translate_one(req.character_a, "Character A") if req.character_a is not None else None
    )
    result_b = (
        _translate_one(req.character_b, "Character B") if req.character_b is not None else None
    )

    return ChatTranslationResponse(character_a=result_a, character_b=result_b)


def save_chat(req: ChatSaveRequest, data_dir: Path, prompt_root: Path) -> ChatSaveResponse:
    """
    Save an in-game chat log session to a timestamped folder under ``data_dir``.
    """
    timestamp = datetime.now(timezone.utc)
    entries_raw = [entry.model_dump() for entry in req.entries]
    log_hash = hashlib.sha256(json.dumps(entries_raw, sort_keys=True).encode()).hexdigest()

    folder_name = save_folder_name(timestamp, log_hash)
    save_dir = data_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    (save_dir / "game_log.md").write_text(
        build_game_log_md(
            entries_raw,
            req.model,
            req.temperature,
            req.max_tokens,
            req.seed,
            timestamp,
        ),
        encoding="utf-8",
    )
    files_written.append("game_log.md")

    for ch, axes in (("a", req.character_a), ("b", req.character_b)):
        if axes:
            (save_dir / f"char_{ch}_payload.json").write_text(
                json.dumps(axes, indent=2), encoding="utf-8"
            )
            files_written.append(f"char_{ch}_payload.json")

    prompt_to_save = req.system_prompt
    if not prompt_to_save:
        try:
            prompt_to_save = load_chat_default_prompt()
        except HTTPException:
            prompt_to_save = None

    if prompt_to_save:
        (save_dir / "system_prompt.md").write_text(
            build_system_prompt_md(prompt_to_save, folder_name),
            encoding="utf-8",
        )
        files_written.append("system_prompt.md")

    if req.system_prompt:
        sp_hash = compute_system_prompt_hash(req.system_prompt)
    else:
        sp_hash = next(
            (
                entry.get("system_prompt_hash")
                for entry in entries_raw
                if entry.get("system_prompt_hash")
            ),
            None,
        )

    # Preserve every distinct prompt used during the live conversation so a
    # saved session can be audited even when the prompt template changed
    # mid-conversation.
    prompt_history_by_key: dict[tuple[str | None, str], dict] = {}
    for index, entry in enumerate(entries_raw, start=1):
        prompt_text = entry.get("system_prompt")
        if not prompt_text:
            continue

        prompt_hash = entry.get("system_prompt_hash")
        if prompt_hash is None:
            prompt_hash = compute_system_prompt_hash(prompt_text)

        key = (prompt_hash, prompt_text)
        history_row = prompt_history_by_key.setdefault(
            key,
            {
                "system_prompt_hash": prompt_hash,
                "system_prompt": prompt_text,
                "entry_indices": [],
            },
        )
        history_row["entry_indices"].append(index)

    system_prompt_history = list(prompt_history_by_key.values())
    for index, history_row in enumerate(system_prompt_history, start=1):
        filename = f"system_prompt_{index:03d}.md"
        (save_dir / filename).write_text(
            build_system_prompt_md(history_row["system_prompt"], folder_name),
            encoding="utf-8",
        )
        files_written.append(filename)
        history_row["filename"] = filename

    per_entry_hashes = [
        {
            "index": i + 1,
            "ch": entry["ch"],
            "input_hash": entry.get("input_hash"),
            "system_prompt_hash": entry.get("system_prompt_hash"),
            "system_prompt": entry.get("system_prompt"),
            "output_hash": entry.get("output_hash"),
            "ipc_id": entry.get("ipc_id"),
            "status": entry.get("status", "success"),
            "error_detail": entry.get("error_detail"),
            "sent_at": entry.get("sent_at"),
            "duration_ms": entry.get("duration_ms"),
        }
        for i, entry in enumerate(entries_raw)
    ]

    metadata = {
        "folder_name": folder_name,
        "timestamp": timestamp.isoformat(),
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "seed": req.seed,
        "entry_count": len(req.entries),
        "has_character_a": req.character_a is not None,
        "has_character_b": req.character_b is not None,
        "character_a_name": req.character_a_name,
        "character_b_name": req.character_b_name,
        "system_prompt_hash": sp_hash,
        "system_prompt_history": system_prompt_history,
        "log_hash": log_hash[:16],
        "per_entry_hashes": per_entry_hashes,
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files_written.append("metadata.json")

    return ChatSaveResponse(
        folder_name=folder_name,
        files=sorted(files_written),
        timestamp=timestamp.isoformat(),
    )
