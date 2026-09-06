# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material variants.

Variants replay regeneration once per material treatment and snapshot each
result, so these tests focus on isolation between variants and on honest
reporting when one variant fails.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from ...service import variants as variants_module

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
    b"\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


_MARKER_PATTERN = re.compile(r"variant-marker:([a-z0-9-]+)")


def _marker_for(config_dict: dict[str, Any]) -> str:
    """Return the marker embedded in the variant prompt for this run."""
    match = _MARKER_PATTERN.search(json.dumps(config_dict))
    return match.group(1) if match else "base-run"


def _merge_pipeline_state(
    session_dir: Path,
    completed_steps: list[str],
    step_outputs: dict[str, Any],
) -> None:
    """Record checkpoint evidence the way the real executor does."""
    state_path = session_dir / "cache" / ".pipeline_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    merged_steps = list(state.get("completed_steps") or [])
    for step in completed_steps:
        if step not in merged_steps:
            merged_steps.append(step)
    merged_outputs = dict(state.get("step_outputs") or {})
    merged_outputs.update(step_outputs)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": merged_steps,
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": merged_outputs,
            }
        )
    )


async def _variant_stub_execute(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    user_email: str = "",
    coverage_policy: str = "allow_partial",
    regeneration_claim: Any | None = None,
) -> None:
    """Minimal claim-aware executor that writes identifiable run artifacts."""
    from ...service.workers.executor import _promote_current_run_artifacts

    manager = session_manager
    session_dir: Path = manager.get_session_dir(session_id)
    marker = _marker_for(config_dict)

    output_dir = session_dir / "output"
    predictions_dir = session_dir / "cache" / "predictions"
    dataset_dir = session_dir / "cache" / "dataset"
    usd_dataset_dir = dataset_dir / "usd"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    usd_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Dataset evidence is produced once and reused by every variant, exactly
    # like a real session that only replays predict/apply/render.
    dataset_path = dataset_dir / "dataset.jsonl"
    if not dataset_path.exists():
        entry = {"id": "/p0", "type": "Mesh", "images": {"composition": "img_0.png"}}
        dataset_path.write_text(json.dumps(entry) + "\n")
        (dataset_dir / "prims.jsonl").write_text(
            json.dumps({"prim_path": "/p0", "type": "Mesh"}) + "\n"
        )
        (usd_dataset_dir / "prims.jsonl").write_text(
            (dataset_dir / "prims.jsonl").read_text()
        )

    predictions_path = predictions_dir / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "/p0", "material": marker, "confidence": 0.9}) + "\n"
    )
    output_usd = output_dir / "scene_with_materials.usd"
    output_usd.write_text(f"#usda 1.0\n# {marker}\n")
    render_path = output_dir / "scene_with_materials.png"
    render_path.write_bytes(_PNG_BYTES + marker.encode("utf-8"))

    completed_steps = [
        "build_dataset_prepare_dataset",
        "build_dataset_usd",
        "predict",
        "apply",
        "render",
    ]
    step_outputs = {
        "build_dataset_usd": {
            "output_dir": str(usd_dataset_dir),
            "usd_dataset_dir": str(usd_dataset_dir),
            "num_prims": 1,
            "num_images": 1,
        },
        "build_dataset_prepare_dataset": {
            "dataset_path": str(dataset_dir),
            "dataset_jsonl_path": str(dataset_path),
            "num_entries": 1,
        },
        "predict": {"predictions_path": str(predictions_path)},
        "apply": {"output_usd_path": str(output_usd)},
        "render": {"rendered_image_path": str(render_path)},
    }
    _merge_pipeline_state(session_dir, completed_steps, step_outputs)
    terminal_updates = {
        "status": "completed",
        "coverage": None,
        "results": {"materials_applied": 1, "marker": marker},
        "completed_at": "1970-01-01T00:00:01Z",
        "can_cancel": False,
        "artifact_validity": {
            "raw_predictions": True,
            "prediction_report": False,
            "restored_predictions": False,
            "applied_output_usd": True,
            "rendered_output_usd": False,
            "final_render": True,
            "cluster_map": False,
            "cluster_report": False,
            "cluster_summary": False,
            "cluster_representatives": False,
            "previews": False,
        },
    }

    if marker == "boom":
        # Mirror the executor's regeneration failure: a durable diagnostic
        # code with no human-readable message, plus the sanitized step-level
        # failure event the listener writes to the session event log.
        assert regeneration_claim is not None
        log_line = json.dumps(
            {
                "session_id": session_id,
                "step": "MaterialRetrieval",
                "state": "failed",
                "message": "RuntimeError during step execution",
                "extra": {"step_name": "apply"},
            }
        )
        with (session_dir / "event_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(log_line + "\n")
        failed = await manager.finalize_regeneration_claim(
            session_id,
            regeneration_claim,
            updates={
                "status": "failed",
                "error_diagnostic": {
                    "code": "material_pipeline_result_failed",
                    "phase": "pipeline_execution",
                    "retryable": False,
                },
                "failed_at": "1970-01-01T00:00:02Z",
                "failed_step": "apply",
                "can_cancel": False,
            },
        )
        assert failed
        return

    if regeneration_claim is None:
        await manager.update_session(session_id, terminal_updates)
        return

    artifact_map: dict[str, str] = {}
    await _promote_current_run_artifacts(
        manager,
        session_id,
        session_dir,
        completed_steps,
        step_outputs,
        regeneration_claim=regeneration_claim,
        artifact_map=artifact_map,
    )
    finalized = await manager.finalize_regeneration_claim(
        session_id,
        regeneration_claim,
        updates=terminal_updates,
        artifact_map=artifact_map,
    )
    assert finalized


@pytest.fixture(autouse=True)
def _claim_aware_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the shared stub with one that honours regeneration claims."""
    from ...service.routers import pipeline_router

    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", _variant_stub_execute, raising=True
    )


async def _create_completed_session(client: Any) -> str:
    """Create a pipeline session and wait for the stub executor to finish."""
    files = {"usd_file": ("scene.usda", b"#usda 1.0\n", "application/octet-stream")}
    create_r = await client.post(
        "/pipeline", files=files, data={"user_email": "test@example.com"}
    )
    assert create_r.status_code == 202
    session_id: str = create_r.json()["session_id"]

    for _ in range(400):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200
        if status_r.json()["status"] == "completed":
            return session_id
        await asyncio.sleep(0.01)
    pytest.fail(f"session {session_id} did not complete")


async def _wait_for_variant_run(client: Any, session_id: str) -> dict[str, Any]:
    """Poll the variants listing until the run leaves the running state."""
    payload: dict[str, Any] = {}
    for _ in range(600):
        response = await client.get(f"/artifacts/{session_id}/variants")
        assert response.status_code == 200
        payload = response.json()
        run = payload.get("run")
        if (
            run
            and run["status"] != "running"
            and not variants_module.variant_run_is_active(session_id)
        ):
            return payload
        await asyncio.sleep(0.02)
    pytest.fail(f"variant run did not finish for {session_id}: {payload}")


@pytest.fixture(autouse=True)
def _fast_variant_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the serial variant loop responsive under test."""
    monkeypatch.setattr(variants_module, "_VARIANT_POLL_SECONDS", 0.02)


@pytest.mark.api
class TestMaterialVariants:
    """Test the serial variant run and its stored artifacts."""

    async def test_variants_are_kept_per_variant(self, client):
        """Each variant keeps its own USD snapshot and echoes its spec."""
        session_id = await _create_completed_session(client)

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "variants": [
                    {"label": "Bare metal", "user_prompt": "variant-marker:bare"},
                    {"label": "Painted", "user_prompt": "variant-marker:painted"},
                ],
            },
        )
        assert response.status_code == 202
        accepted = response.json()
        assert accepted["total"] == 2
        assert len(accepted["variant_ids"]) == 2

        payload = await _wait_for_variant_run(client, session_id)
        assert payload["run"]["status"] == "completed", json.dumps(
            payload["variants"], indent=2
        )
        assert payload["run"]["completed_count"] == 2
        assert payload["total"] == 2

        labels = [variant["label"] for variant in payload["variants"]]
        assert labels == ["Bare metal", "Painted"]

        bodies: list[bytes] = []
        previews: list[bytes] = []
        for variant in payload["variants"]:
            assert variant["status"] == "completed"
            assert variant["outcome"]["status"] == "completed"
            assert variant["spec"]["steps"] == [
                "build_dataset_prepare_dataset",
                "predict",
                "apply",
                "render",
            ]
            assert variant["usd_url"] is not None
            usd_r = await client.get(variant["usd_url"])
            assert usd_r.status_code == 200
            bodies.append(usd_r.content)

            assert variant["preview_url"] is not None
            preview_r = await client.get(variant["preview_url"])
            assert preview_r.status_code == 200
            assert preview_r.headers["content-type"].startswith("image/png")
            previews.append(preview_r.content)

        # Snapshots are per-variant, so the second run cannot overwrite the
        # first one's artifacts.
        first, second = payload["variants"]
        assert first["variant_id"] != second["variant_id"]
        assert b"variant-marker:bare" not in bodies[0]
        assert b"bare" in bodies[0]
        assert b"painted" in bodies[1]
        assert previews[0] != previews[1]

    async def test_partial_failure_still_returns_other_variants(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A rejected variant is reported without losing its siblings."""
        from ...service.routers import pipeline_router

        session_id = await _create_completed_session(client)
        real_regenerate = pipeline_router.regenerate_pipeline

        async def flaky_regenerate(inner_session_id: str, request: Any):
            if request.user_prompt == "fail-me":
                raise HTTPException(status_code=400, detail="simulated rejection")
            return await real_regenerate(inner_session_id, request)

        monkeypatch.setattr(
            pipeline_router, "regenerate_pipeline", flaky_regenerate, raising=True
        )

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "steps": ["build_dataset_prepare_dataset", "predict", "apply"],
                "variants": [
                    {"label": "First", "user_prompt": "variant-marker:first"},
                    {"label": "Broken", "user_prompt": "fail-me"},
                    {"label": "Third", "user_prompt": "variant-marker:third"},
                ],
            },
        )
        assert response.status_code == 202

        payload = await _wait_for_variant_run(client, session_id)
        assert payload["run"]["status"] == "completed"
        assert payload["run"]["failed_count"] == 1
        assert payload["run"]["completed_count"] == 2

        by_label = {variant["label"]: variant for variant in payload["variants"]}
        assert by_label["First"]["status"] == "completed"
        assert by_label["Third"]["status"] == "completed"
        assert by_label["First"]["usd_url"] is not None
        assert by_label["Third"]["usd_url"] is not None

        broken = by_label["Broken"]
        assert broken["status"] == "failed"
        assert broken["usd_url"] is None
        assert "simulated rejection" in broken["outcome"]["error"]

    async def test_failed_variant_reports_a_reason(self, client):
        """A pipeline failure surfaces its diagnostic, not a bare 'failed'."""
        session_id = await _create_completed_session(client)

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "variants": [
                    {"label": "Good", "user_prompt": "variant-marker:good"},
                    {"label": "Doomed", "user_prompt": "variant-marker:boom"},
                ]
            },
        )
        assert response.status_code == 202

        payload = await _wait_for_variant_run(client, session_id)
        by_label = {variant["label"]: variant for variant in payload["variants"]}

        assert by_label["Good"]["status"] == "completed"
        doomed = by_label["Doomed"]
        assert doomed["status"] == "failed"
        assert doomed["outcome"]["error"]
        assert "material_pipeline_result_failed" in doomed["outcome"]["error"]
        assert doomed["outcome"]["failed_step"] == "apply"
        assert doomed["outcome"]["error_diagnostic"]["phase"] == "pipeline_execution"
        assert "failed during 'apply'" in doomed["outcome"]["error"]

    async def test_variant_run_is_reported_on_status(self, client):
        """The session status names the variant that is currently running."""
        session_id = await _create_completed_session(client)

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "steps": ["build_dataset_prepare_dataset", "predict", "apply"],
                "variants": [
                    {"label": "Only one", "user_prompt": "variant-marker:only"}
                ],
            },
        )
        assert response.status_code == 202

        observed_labels: set[str] = set()
        for _ in range(600):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            assert status_r.status_code == 200
            variant_run = status_r.json().get("variant_run")
            if variant_run:
                if variant_run.get("current_label"):
                    observed_labels.add(variant_run["current_label"])
                if variant_run["status"] != "running":
                    break
            await asyncio.sleep(0.02)

        await _wait_for_variant_run(client, session_id)
        assert "Only one" in observed_labels

    async def test_reset_discards_previous_variants(self, client):
        """A reset run replaces the stored variants instead of appending."""
        session_id = await _create_completed_session(client)

        first = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "steps": ["build_dataset_prepare_dataset", "predict", "apply"],
                "variants": [
                    {"label": "Original", "user_prompt": "variant-marker:original"}
                ],
            },
        )
        assert first.status_code == 202
        await _wait_for_variant_run(client, session_id)

        second = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "steps": ["build_dataset_prepare_dataset", "predict", "apply"],
                "reset": True,
                "variants": [
                    {
                        "label": "Replacement",
                        "user_prompt": "variant-marker:replacement",
                    }
                ],
            },
        )
        assert second.status_code == 202
        payload = await _wait_for_variant_run(client, session_id)

        assert payload["total"] == 1
        assert payload["variants"][0]["label"] == "Replacement"

    async def test_rejects_unknown_material_library(self, client):
        """An unresolvable library fails the whole request up front."""
        session_id = await _create_completed_session(client)

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "variants": [
                    {"label": "Bad library", "material_library": "not-a-library"}
                ]
            },
        )
        assert response.status_code == 400
        assert "not-a-library" in response.json()["detail"]

    async def test_rejects_layer_only_without_apply(self, client):
        """layer_only needs the apply step, mirroring regeneration."""
        session_id = await _create_completed_session(client)

        response = await client.post(
            f"/pipeline/{session_id}/variants",
            json={
                "steps": ["predict"],
                "variants": [{"label": "Layer", "layer_only": True}],
            },
        )
        assert response.status_code == 400

    async def test_variants_listing_for_unknown_session(self, client):
        """Listing an unknown session is a 404, not an empty grid."""
        response = await client.get(
            "/artifacts/11111111-2222-3333-4444-555555555555/variants"
        )
        assert response.status_code == 404

    async def test_invalid_variant_id_is_rejected(self, client):
        """Variant identifiers are validated before touching the store."""
        session_id = await _create_completed_session(client)
        response = await client.get(
            f"/artifacts/{session_id}/variants/..%2F..%2Fetc/preview"
        )
        assert response.status_code in {400, 404}
