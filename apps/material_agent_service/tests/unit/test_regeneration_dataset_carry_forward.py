# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendered dataset views must survive a prompt-only regeneration.

``build_dataset_prepare_dataset`` re-emits the dataset records but never
re-renders ``cache/dataset/usd/``.  If those views were dropped from a
regeneration's publication map, the next regeneration would find no reusable
render evidence and hydration would delete the local copies, which makes a
session single-use for prompt changes and for material variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from ...service.artifact_lineage import initial_artifact_validity
from ...service.session.manager import SessionManager
from ...service.workers import executor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_regeneration_keeps_rendered_dataset_views(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)

    view_key = "cache/dataset/usd/prim_0/view_0.png"
    records_key = "cache/dataset/dataset.jsonl"
    await manager.put_bytes_to_store(session_id, view_key, b"rendered-view")
    await manager.put_bytes_to_store(session_id, records_key, b"{}\n")
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "artifact_validity": initial_artifact_validity(),
        },
    )

    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=snapshot.version,
        lease_seconds=120.0,
    )
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None

    artifact_map: dict[str, str] = {}
    await executor._carry_forward_regeneration_artifacts(
        manager,
        session_id,
        claim,
        metadata,
        initial_artifact_validity(),
        artifact_map,
        {"build_dataset_prepare_dataset", "predict", "apply", "render"},
    )

    assert view_key in artifact_map
    assert artifact_map[view_key].startswith(f"{claim.artifact_prefix}/")
    assert (
        await manager.read_from_store(
            session_id,
            artifact_map[view_key],
        )
        == b"rendered-view"
    )
    # The dataset records themselves are re-emitted by the step that ran.
    assert records_key not in artifact_map


@pytest.mark.unit
def test_regeneration_checkpoint_keeps_upstream_completions(tmp_path: Path) -> None:
    """The run's checkpoint must retain evidence for steps it did not re-run."""
    state_path = tmp_path / "cache" / ".pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": ["optimize_usd", "build_dataset_usd"],
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": {"build_dataset_usd": {"num_images": 20}},
            }
        ),
        encoding="utf-8",
    )
    carried = executor._checkpoint_completed_steps(tmp_path)
    assert carried == ["optimize_usd", "build_dataset_usd"]

    # The pipeline rewrites the checkpoint with only the steps it executed.
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": ["build_dataset_prepare_dataset", "predict"],
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": {"predict": {"predictions_path": "p.jsonl"}},
            }
        ),
        encoding="utf-8",
    )
    executor._restore_carried_checkpoint_steps(tmp_path, carried)

    merged = json.loads(state_path.read_text(encoding="utf-8"))["completed_steps"]
    assert merged == [
        "optimize_usd",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
    ]


@pytest.mark.unit
def test_regeneration_checkpoint_does_not_revive_failed_steps(tmp_path: Path) -> None:
    """A step that failed in this run stays out of the merged evidence."""
    state_path = tmp_path / "cache" / ".pipeline_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": ["build_dataset_prepare_dataset"],
                "failed_steps": ["predict"],
                "step_errors": {"predict": "boom"},
                "step_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    executor._restore_carried_checkpoint_steps(tmp_path, ["predict", "optimize_usd"])

    merged = json.loads(state_path.read_text(encoding="utf-8"))["completed_steps"]
    assert "predict" not in merged
    assert merged == ["optimize_usd", "build_dataset_prepare_dataset"]
