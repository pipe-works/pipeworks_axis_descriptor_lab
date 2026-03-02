"""
Save/export orchestration helpers for the character-description page.

This module contains the filesystem-writing logic that was previously
embedded in the ``POST /api/save`` route handler. The route layer now owns
HTTP concerns, while this module owns deterministic save-folder creation and
file generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from pipeworks_ipc import (
    compute_ipc_id,
    compute_output_hash,
    compute_system_prompt_hash,
    payload_hash,
)

from app.save_formatting import (
    build_baseline_md,
    build_output_md,
    build_system_prompt_md,
    save_folder_name,
)
from app.save_package import build_manifest
from app.schema import SaveRequest, SaveResponse
from app.signal_isolation import compute_delta


def save_run(req: SaveRequest, data_dir: Path) -> SaveResponse:
    """
    Persist all session state to individual files inside a new subfolder of
    ``data_dir`` and return the resulting ``SaveResponse``.

    The implementation intentionally mirrors the previous ``app.main.save_run``
    behavior so the save format, provenance hashes, and conditional files stay
    byte-for-byte compatible with the pre-refactor endpoint.
    """
    now = datetime.now(timezone.utc)
    input_hash = payload_hash(req.payload)
    folder_name = save_folder_name(now, input_hash)
    save_dir = data_dir / folder_name

    sp_hash = compute_system_prompt_hash(req.system_prompt)
    out_hash = compute_output_hash(req.output) if req.output is not None else None
    ipc: str | None = None
    if req.output is not None:
        ipc = compute_ipc_id(
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.payload.seed,
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    files_written: list[str] = []

    try:
        (save_dir / "payload.json").write_text(
            json.dumps(req.payload.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("payload.json")

        system_prompt_md = build_system_prompt_md(req.system_prompt, folder_name)
        (save_dir / "system_prompt.md").write_text(system_prompt_md, encoding="utf-8")
        files_written.append("system_prompt.md")

        if req.output is not None:
            output_md = build_output_md(
                req.output,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                seed=req.payload.seed,
                timestamp=now,
                input_hash=input_hash,
                system_prompt_hash=sp_hash,
                ipc_id=ipc,
            )
            (save_dir / "output.md").write_text(output_md, encoding="utf-8")
            files_written.append("output.md")

        if req.baseline is not None:
            baseline_md = build_baseline_md(req.baseline, folder_name)
            (save_dir / "baseline.md").write_text(baseline_md, encoding="utf-8")
            files_written.append("baseline.md")

        if req.output is not None and req.baseline is not None:
            removed, added = compute_delta(req.baseline, req.output)
            delta_data = {
                "removed": removed,
                "added": added,
                "removed_count": len(removed),
                "added_count": len(added),
            }
            (save_dir / "delta.json").write_text(
                json.dumps(delta_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files_written.append("delta.json")

        if req.transformation_map is not None and len(req.transformation_map) > 0:
            tmap_data = {
                "rows": [row.model_dump() for row in req.transformation_map],
                "row_count": len(req.transformation_map),
            }
            (save_dir / "transformation_map.json").write_text(
                json.dumps(tmap_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files_written.append("transformation_map.json")

        metadata: dict[str, object] = {
            "folder_name": folder_name,
            "timestamp": now.isoformat(),
            "input_hash": input_hash,
            "system_prompt_hash": sp_hash,
            "output_hash": out_hash,
            "ipc_id": ipc,
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "seed": req.payload.seed,
            "world_id": req.payload.world_id,
            "policy_hash": req.payload.policy_hash,
            "axis_count": len(req.payload.axes),
            "payload_name": req.payload_name,
            "diff_change_pct": req.diff_change_pct,
        }

        metadata["manifest"] = build_manifest(save_dir, files_written)

        (save_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("metadata.json")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write save files to {save_dir}: {exc}",
        ) from exc

    return SaveResponse(
        folder_name=folder_name,
        files=sorted(files_written),
        input_hash=input_hash,
        timestamp=now.isoformat(),
        system_prompt_hash=sp_hash,
        output_hash=out_hash,
        ipc_id=ipc,
    )
