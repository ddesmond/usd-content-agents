# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material variants: several material treatments of one already-built asset.

A variant run replays the regeneration machinery once per requested treatment
and snapshots the result before the next run overwrites the mutable session
output.  Regeneration is deliberately destructive - it reuses the same session
directory and rewrites ``output/scene_with_materials.usd`` and the final render
- so a variant only survives because its artifacts are copied into an immutable
``variants/<variant_id>/`` prefix as soon as the run reaches a terminal state.

The snapshots are intentionally read through the same publication resolution
the artifact endpoints use, so a variant captures the bytes that the completed
run actually published rather than whatever happens to be on local disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .session.manager import RegenerationClaimConflictError, SessionManager

logger = logging.getLogger(__name__)

VARIANTS_METADATA_KEY = "material_variants"
VARIANTS_DIR_NAME = "variants"
MAX_VARIANTS_PER_RUN = 8
MAX_STORED_VARIANTS = 64

_VARIANT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_COPY_CHUNK_SIZE = 1024 * 1024
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_VARIANT_POLL_SECONDS = 1.0
_VARIANT_TIMEOUT_SECONDS = 3600.0

# Logical artifact name -> (canonical session key, snapshot file name).
_SNAPSHOT_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("final_render", "output/scene_with_materials.png", "final_render.png"),
    ("output_usd", "output/scene_with_materials.usd", "output.usd"),
    ("output_usd_flat", "output/scene_with_materials_flat.usd", "output_flat.usd"),
    ("predictions", "cache/predictions/predictions.jsonl", "predictions.jsonl"),
    (
        "restored_predictions",
        "cache/restored/restored_predictions.jsonl",
        "restored_predictions.jsonl",
    ),
)

_ARTIFACT_MEDIA_TYPES = {
    "final_render.png": "image/png",
    "output.usd": "application/octet-stream",
    "output_flat.usd": "application/octet-stream",
    "predictions.jsonl": "application/x-ndjson",
    "restored_predictions.jsonl": "application/x-ndjson",
}


class VariantRunError(RuntimeError):
    """Raised when a variant run cannot continue past a shared-state failure."""


# ---------------------------------------------------------------------------
# In-memory progress for the active run
# ---------------------------------------------------------------------------

# Variant runs are orchestrated in-process and the durable block is only
# rewritten between variants (session metadata is fenced while a regeneration
# claim is live).  This mirror keeps ``/pipeline/{id}/status`` able to report
# which variant is executing without an extra disk read per poll.
_active_runs: dict[str, dict[str, Any]] = {}
_active_tasks: dict[str, asyncio.Task[None]] = {}


def active_variant_run(session_id: str) -> dict[str, Any] | None:
    """Return live run progress for a session, or ``None`` when idle."""
    run = _active_runs.get(session_id)
    return dict(run) if run else None


def variant_run_is_active(session_id: str) -> bool:
    """Return whether a variant run currently owns this session."""
    task = _active_tasks.get(session_id)
    return task is not None and not task.done()


def register_variant_run(session_id: str, task: asyncio.Task[None]) -> None:
    """Hold a strong reference to the orchestrator task."""
    _active_tasks[session_id] = task
    task.add_done_callback(lambda _task: _release_variant_run(session_id, _task))


def _release_variant_run(session_id: str, task: asyncio.Task[None]) -> None:
    if _active_tasks.get(session_id) is task:
        _active_tasks.pop(session_id, None)
    _active_runs.pop(session_id, None)


# ---------------------------------------------------------------------------
# Durable variant block
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def variants_block(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the persisted variant block with a stable shape."""
    block = (metadata or {}).get(VARIANTS_METADATA_KEY)
    if not isinstance(block, Mapping):
        return {"run": None, "variants": []}
    run = block.get("run")
    entries = block.get("variants")
    return {
        "run": dict(run) if isinstance(run, Mapping) else None,
        "variants": [dict(entry) for entry in entries if isinstance(entry, Mapping)]
        if isinstance(entries, list)
        else [],
    }


def new_variant_id(index: int) -> str:
    """Return a filesystem- and URL-safe identifier for one variant."""
    return f"v{index + 1:02d}-{uuid.uuid4().hex[:8]}"


def validate_variant_id(variant_id: str) -> bool:
    """Return whether an identifier is safe to resolve against the store."""
    return bool(_VARIANT_ID_PATTERN.match(variant_id))


def variant_artifact_key(variant_id: str, file_name: str) -> str:
    """Return the immutable store key for one snapshotted variant artifact."""
    return f"{VARIANTS_DIR_NAME}/{variant_id}/{file_name}"


def variant_media_type(file_name: str) -> str:
    """Return the response media type for a snapshotted variant artifact."""
    return _ARTIFACT_MEDIA_TYPES.get(file_name, "application/octet-stream")


def find_variant(
    metadata: Mapping[str, Any] | None,
    variant_id: str,
) -> dict[str, Any] | None:
    """Return one stored variant entry by identifier."""
    for entry in variants_block(metadata)["variants"]:
        if entry.get("variant_id") == variant_id:
            return entry
    return None


def variant_urls(session_id: str, entry: Mapping[str, Any]) -> dict[str, str | None]:
    """Return browser-consumable URLs for one variant entry."""
    variant_id = entry.get("variant_id")
    artifacts = entry.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    base = f"/artifacts/{session_id}/variants/{variant_id}"
    has_usd = bool(artifacts.get("output_usd") or artifacts.get("output_usd_flat"))
    has_predictions = bool(
        artifacts.get("restored_predictions") or artifacts.get("predictions")
    )
    return {
        "preview_url": f"{base}/preview" if artifacts.get("final_render") else None,
        "usd_url": f"{base}/output" if has_usd else None,
        "predictions_url": f"{base}/predictions" if has_predictions else None,
    }


async def persist_block(
    manager: SessionManager,
    session_id: str,
    block: Mapping[str, Any],
    *,
    sync_files: bool = False,
) -> None:
    """Write the variant block into durable session metadata."""
    try:
        await manager.update_session(
            session_id,
            {VARIANTS_METADATA_KEY: dict(block)},
            sync_files=sync_files,
        )
    except RegenerationClaimConflictError as exc:
        raise VariantRunError(
            "Session is fenced by an active regeneration claim; "
            "variant progress cannot be recorded"
        ) from exc


# ---------------------------------------------------------------------------
# Snapshotting
# ---------------------------------------------------------------------------


async def _copy_session_artifact(
    manager: SessionManager,
    session_id: str,
    metadata: Mapping[str, Any],
    canonical_key: str,
    destination: Path,
) -> bool:
    """Copy one published run artifact into the variant snapshot directory."""
    session_dir = manager.get_session_dir(session_id)
    store_key = manager.resolve_published_artifact_key(
        metadata,
        canonical_key,
        legacy_key=canonical_key,
    )
    pending = destination.with_name(f".{destination.name}.partial")
    if store_key is not None:
        stream = await manager.iter_store_chunks(
            session_id,
            store_key,
            chunk_size=_COPY_CHUNK_SIZE,
        )
        if stream is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with pending.open("wb") as handle:
                async for chunk in stream:
                    await asyncio.to_thread(handle.write, chunk)
            pending.replace(destination)
            return True

    # Sessions without a publication map keep canonical bytes on local disk.
    if manager.store.kind != "local" or metadata.get("published_artifacts") is not None:
        return False
    artifact = await manager.open_local_artifact(
        session_id,
        session_dir / canonical_key,
    )
    if artifact is None:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pending.open("wb") as handle:
            await asyncio.to_thread(shutil.copyfileobj, artifact.stream, handle)
    finally:
        artifact.stream.close()
    pending.replace(destination)
    return True


async def snapshot_variant_artifacts(
    manager: SessionManager,
    session_id: str,
    variant_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Freeze the current run output under ``variants/<variant_id>/``.

    Returns the logical artifact names that were captured, mapped to their
    immutable store keys.  Missing artifacts are skipped rather than failing:
    a partially successful variant still deserves whatever it produced.
    """
    session_dir = manager.get_session_dir(session_id)
    variant_dir = session_dir / VARIANTS_DIR_NAME / variant_id
    await asyncio.to_thread(variant_dir.mkdir, parents=True, exist_ok=True)

    captured: dict[str, str] = {}
    for artifact_name, canonical_key, file_name in _SNAPSHOT_ARTIFACTS:
        destination = variant_dir / file_name
        try:
            copied = await _copy_session_artifact(
                manager,
                session_id,
                metadata,
                canonical_key,
                destination,
            )
        except (OSError, RuntimeError, ValueError):
            logger.exception(
                "Failed to snapshot %s for variant %s",
                canonical_key,
                variant_id,
            )
            continue
        if not copied:
            continue
        key = variant_artifact_key(variant_id, file_name)
        captured[artifact_name] = key
        if manager.store.kind != "local":
            await manager.put_file_to_store(
                session_id,
                key,
                str(destination),
                _ARTIFACT_MEDIA_TYPES.get(file_name),
            )
    return captured


async def discard_variants(manager: SessionManager, session_id: str) -> None:
    """Remove every stored variant snapshot for a session."""
    session_dir = manager.get_session_dir(session_id)
    variants_dir = session_dir / VARIANTS_DIR_NAME
    await asyncio.to_thread(shutil.rmtree, variants_dir, True)
    if manager.store.kind != "local":
        keys = await manager.store.list_keys(
            session_id,
            prefix=f"{VARIANTS_DIR_NAME}/",
        )
        for key in keys:
            await manager.store.delete_file(session_id, key)


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------


_EVENT_LOG_TAIL_BYTES = 64 * 1024


async def _run_failure_context(
    manager: SessionManager,
    session_id: str,
) -> dict[str, str]:
    """Return the last step-level failure the run recorded in its event log.

    The executor persists only a durable diagnostic code for a failed
    regeneration.  The sanitized event log additionally names the step and task
    that failed, which is what makes a failed variant actionable in the UI.
    """
    log_path = manager.get_session_dir(session_id) / "event_log.jsonl"

    def read_tail() -> list[str]:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - _EVENT_LOG_TAIL_BYTES))
                return handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []

    lines = await asyncio.to_thread(read_tail)
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("state") != "failed":
            continue
        if event.get("step") == "pipeline":
            continue
        extra = event.get("extra")
        extra = extra if isinstance(extra, Mapping) else {}
        context: dict[str, str] = {}
        step = extra.get("step_name") or event.get("step")
        if isinstance(step, str) and step:
            context["failed_step"] = step
        task = event.get("step")
        if isinstance(task, str) and task and task != context.get("failed_step"):
            context["failed_task"] = task
        message = event.get("message")
        if isinstance(message, str) and message:
            context["message"] = message
        return context
    return {}


def _failure_reason(
    metadata: Mapping[str, Any],
    status: str,
    failure_context: Mapping[str, str] | None = None,
) -> str | None:
    """Return the most specific failure reason the run left behind.

    A failed run persists either a human-readable message or a durable
    diagnostic code.  Reporting "failed" with an empty reason is not
    actionable, so fall back through every diagnostic the executor writes
    before giving up on an explicit explanation.
    """
    if status == "completed":
        return None
    context = dict(failure_context or {})
    reasons: list[str] = []
    error = metadata.get("error")
    if isinstance(error, str) and error.strip():
        reasons.append(error)
    else:
        diagnostic = metadata.get("error_diagnostic")
        if isinstance(diagnostic, Mapping):
            code = diagnostic.get("code")
            phase = diagnostic.get("phase")
            if isinstance(code, str) and code:
                reasons.append(f"{code} (phase: {phase})" if phase else code)

    step = context.get("failed_step") or metadata.get("failed_step")
    if isinstance(step, str) and step:
        task = context.get("failed_task")
        where = f"failed during '{step}'"
        if task:
            where += f" ({task})"
        message = context.get("message")
        reasons.append(f"{where}: {message}" if message else where)

    if reasons:
        return " - ".join(reasons)
    if status == "cancelled":
        return "Pipeline run was cancelled"
    return "Pipeline reported failure without a diagnostic message"


def _variant_outcome(
    metadata: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    duration_seconds: float,
    failure_context: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize how one regeneration turned out for the variant listing."""
    status = (
        "completed"
        if metadata.get("status") == "completed"
        else str(metadata.get("status") or "failed")
    )
    results = metadata.get("results")
    coverage = metadata.get("coverage")
    diagnostic = metadata.get("error_diagnostic")
    stats = dict(results) if isinstance(results, Mapping) else {}
    return {
        "status": status,
        "error": _failure_reason(metadata, status, failure_context),
        "error_diagnostic": (
            dict(diagnostic) if isinstance(diagnostic, Mapping) else None
        ),
        "failed_step": metadata.get("failed_step")
        or (failure_context or {}).get("failed_step"),
        "duration_seconds": round(duration_seconds, 2),
        "coverage": dict(coverage) if isinstance(coverage, Mapping) else None,
        "readiness_grade": (
            coverage.get("readiness_grade") if isinstance(coverage, Mapping) else None
        ),
        "stats": stats,
        "material_library": spec.get("material_library"),
        "material_profile": spec.get("material_profile"),
        "layer_only": bool(spec.get("layer_only")),
    }


def _pending_outcome(
    spec: Mapping[str, Any], status: str = "pending"
) -> dict[str, Any]:
    return {
        "status": status,
        "error": None,
        "error_diagnostic": None,
        "failed_step": None,
        "duration_seconds": None,
        "coverage": None,
        "readiness_grade": None,
        "stats": {},
        "material_library": spec.get("material_library"),
        "material_profile": spec.get("material_profile"),
        "layer_only": bool(spec.get("layer_only")),
    }


def plan_variant_entry(
    *,
    run_id: str,
    index: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the durable entry for one queued variant."""
    return {
        "variant_id": new_variant_id(index),
        "run_id": run_id,
        "index": index,
        "label": str(spec["label"]),
        "status": "pending",
        "spec": dict(spec),
        "outcome": _pending_outcome(spec),
        "artifacts": {},
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
    }


def _failed_outcome(
    spec: Mapping[str, Any],
    reason: str,
    *,
    duration_seconds: float,
    failed_step: str | None = None,
    status: str = "failed",
) -> dict[str, Any]:
    return {
        "status": status,
        "error": reason,
        "error_diagnostic": None,
        "failed_step": failed_step,
        "duration_seconds": round(duration_seconds, 2),
        "coverage": None,
        "readiness_grade": None,
        "stats": {},
        "material_library": spec.get("material_library"),
        "material_profile": spec.get("material_profile"),
        "layer_only": bool(spec.get("layer_only")),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _await_terminal_metadata(
    manager: SessionManager,
    session_id: str,
    *,
    timeout_seconds: float = _VARIANT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wait until a regeneration is finished and its metadata is authoritative."""
    from .routers.pipeline_router import _terminal_metadata_ready
    from .runtime import get_job_registry

    job_registry = get_job_registry()
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        metadata = await manager.get_session_metadata(session_id)
        if metadata is None:
            raise VariantRunError("Session disappeared during the variant run")
        if (
            metadata.get("status") in _TERMINAL_STATUSES
            and _terminal_metadata_ready(metadata)
            and metadata.get("terminal_events_quiesced") is not False
            and not job_registry.is_running(session_id)
        ):
            return metadata
        if asyncio.get_running_loop().time() > deadline:
            raise VariantRunError(
                f"Variant did not reach a terminal state within {int(timeout_seconds)}s"
            )
        await asyncio.sleep(_VARIANT_POLL_SECONDS)


def _run_progress(
    run_id: str,
    *,
    status: str,
    total: int,
    started_at: str,
    entries: Iterable[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
    completed_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    run_entries = [entry for entry in entries if entry.get("run_id") == run_id]
    return {
        "run_id": run_id,
        "status": status,
        "total": total,
        "started_at": started_at,
        "completed_at": completed_at,
        "completed_count": sum(
            1 for entry in run_entries if entry.get("status") == "completed"
        ),
        "failed_count": sum(
            1 for entry in run_entries if entry.get("status") == "failed"
        ),
        "current_index": current.get("index") if current else None,
        "current_variant_id": current.get("variant_id") if current else None,
        "current_label": current.get("label") if current else None,
        "error": error,
    }


async def execute_variant_run(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    planned: list[dict[str, Any]],
) -> None:
    """Run every planned variant serially, snapshotting each result.

    Each variant is executed through the existing regeneration endpoint so the
    cached dataset and multi-view renders are reused.  One variant failing does
    not abort the run: the failure is recorded on its entry and the next
    variant starts.
    """
    from .models.requests import RegenerateRequest
    from .routers.pipeline_router import regenerate_pipeline

    started_at = _now()
    metadata = await manager.get_session_metadata(session_id)
    block = variants_block(metadata)
    entries: list[dict[str, Any]] = block["variants"]
    entries.extend(planned)
    total = len(planned)

    def publishable_block(
        status: str,
        current: Mapping[str, Any] | None,
        *,
        completed_at: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        trimmed = entries[-MAX_STORED_VARIANTS:]
        progress = _run_progress(
            run_id,
            status=status,
            total=total,
            started_at=started_at,
            entries=trimmed,
            current=current,
            completed_at=completed_at,
            error=error,
        )
        if status == "running":
            _active_runs[session_id] = progress
        else:
            _active_runs.pop(session_id, None)
        return {"run": progress, "variants": trimmed}

    run_error: str | None = None
    try:
        for entry in planned:
            spec = entry["spec"]
            entry["status"] = "running"
            entry["outcome"] = _pending_outcome(spec, status="running")
            entry["started_at"] = _now()
            await persist_block(
                manager,
                session_id,
                publishable_block("running", entry),
            )
            logger.info(
                "Variant %s/%s starting for %s: %s",
                entry["index"] + 1,
                total,
                session_id[:8],
                entry["label"],
            )
            variant_started = asyncio.get_running_loop().time()
            try:
                await regenerate_pipeline(
                    session_id,
                    RegenerateRequest(
                        steps=spec["steps"],
                        user_prompt=spec.get("user_prompt"),
                        layer_only=bool(spec.get("layer_only")),
                        coverage_policy=spec.get("coverage_policy"),
                        material_library=spec.get("material_library"),
                        material_profile=spec.get("material_profile"),
                    ),
                )
            except HTTPException as exc:
                elapsed = asyncio.get_running_loop().time() - variant_started
                entry["status"] = "failed"
                entry["completed_at"] = _now()
                entry["outcome"] = _failed_outcome(
                    spec,
                    f"Regeneration was rejected ({exc.status_code}): {exc.detail}",
                    duration_seconds=elapsed,
                    failed_step="regeneration_admission",
                )
                logger.warning(
                    "Variant %s/%s finished for %s: %s (rejected, %.1fs): %s",
                    entry["index"] + 1,
                    total,
                    session_id[:8],
                    entry["label"],
                    elapsed,
                    exc.detail,
                )
                await persist_block(
                    manager,
                    session_id,
                    publishable_block("running", None),
                )
                continue

            terminal = await _await_terminal_metadata(manager, session_id)
            elapsed = asyncio.get_running_loop().time() - variant_started
            captured = await snapshot_variant_artifacts(
                manager,
                session_id,
                entry["variant_id"],
                terminal,
            )
            entry["artifacts"] = captured
            entry["completed_at"] = _now()
            failure_context = (
                await _run_failure_context(manager, session_id)
                if terminal.get("status") != "completed"
                else {}
            )
            entry["outcome"] = _variant_outcome(
                terminal,
                spec,
                duration_seconds=elapsed,
                failure_context=failure_context,
            )
            entry["status"] = entry["outcome"]["status"]
            if entry["status"] == "completed" and "final_render" not in captured:
                entry["outcome"]["error"] = (
                    "Variant completed but produced no final render image"
                )
            logger.info(
                "Variant %s/%s finished for %s: %s (%s, %.1fs)%s",
                entry["index"] + 1,
                total,
                session_id[:8],
                entry["label"],
                entry["status"],
                elapsed,
                f": {entry['outcome']['error']}"
                if entry["outcome"].get("error")
                else "",
            )
            await persist_block(
                manager,
                session_id,
                publishable_block("running", None),
                sync_files=True,
            )
            if entry["status"] == "cancelled":
                # Cancelling the running pipeline cancels the whole run: the
                # remaining variants would only rerun the cancelled work.
                for pending in planned:
                    if pending["status"] == "pending":
                        pending["status"] = "cancelled"
                        pending["outcome"] = _failed_outcome(
                            pending["spec"],
                            "Variant run was cancelled",
                            duration_seconds=0.0,
                            status="cancelled",
                        )
                await _persist_quietly(
                    manager,
                    session_id,
                    publishable_block(
                        "cancelled",
                        None,
                        completed_at=_now(),
                        error="Variant run was cancelled",
                    ),
                )
                return
    except asyncio.CancelledError:
        for entry in planned:
            if entry["status"] in {"pending", "running"}:
                entry["status"] = "cancelled"
                entry["outcome"] = _failed_outcome(
                    entry["spec"],
                    "Variant run was cancelled",
                    duration_seconds=0.0,
                    status="cancelled",
                )
        with_cancel = publishable_block(
            "cancelled",
            None,
            completed_at=_now(),
            error="Variant run was cancelled",
        )
        await _persist_quietly(manager, session_id, with_cancel)
        raise
    except Exception as exc:  # noqa: BLE001 - recorded on the durable run block
        logger.exception("Variant run failed for %s", session_id)
        run_error = str(exc)
        for entry in planned:
            if entry["status"] in {"pending", "running"}:
                entry["status"] = "failed"
                entry["outcome"] = _failed_outcome(
                    entry["spec"],
                    run_error,
                    duration_seconds=0.0,
                )
        await _persist_quietly(
            manager,
            session_id,
            publishable_block("failed", None, completed_at=_now(), error=run_error),
        )
        return

    failures = sum(1 for entry in planned if entry["status"] != "completed")
    status = "completed" if failures < total else "failed"
    await _persist_quietly(
        manager,
        session_id,
        publishable_block(
            status,
            None,
            completed_at=_now(),
            error=None if status == "completed" else "Every variant failed",
        ),
        sync_files=True,
    )


async def _persist_quietly(
    manager: SessionManager,
    session_id: str,
    block: Mapping[str, Any],
    *,
    sync_files: bool = False,
) -> None:
    """Persist the terminal block without masking the original failure."""
    try:
        await persist_block(manager, session_id, block, sync_files=sync_files)
    except Exception:  # noqa: BLE001 - terminal bookkeeping must not mask failures
        logger.exception("Failed to persist terminal variant block for %s", session_id)
