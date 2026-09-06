# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Material Agent Service API."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class PipelineStep(StrEnum):
    """Available pipeline steps."""

    BUILD_DATASET_PREPARE_DATASET = "build_dataset_prepare_dataset"
    BUILD_DATASET = "build_dataset_usd"
    CLUSTER_PRIMS = "cluster_prims"
    EXPAND_CLUSTER_PREDICTIONS = "expand_cluster_predictions"
    PREDICT = "predict"
    APPLY = "apply"
    RENDER = "render"


class PipelineRequest(BaseModel):
    """Simplified pipeline request for MVP.

    All materials, VLM, LLM configs are pre-configured on backend.
    User provides USD file, optional prompt, and optional reference images.
    """

    # Execution control
    steps: list[PipelineStep] | None = Field(
        default=None,
        description="Steps to execute. If None, runs all steps.",
        examples=[["build_dataset_usd", "predict", "apply"]],
    )

    # Rendering options
    camera_views: list[str] = Field(
        default=["+x+y+z", "-x-y-z"],
        description="Camera positions for rendering",
        examples=[["+x+y+z", "-x+y+z", "+x-y+z", "-x-y+z"]],
    )

    # User-configurable prompt
    user_prompt: str | None = Field(
        default=None,
        description="Custom user prompt for VLM. If None, uses backend default.",
        examples=[
            "Please identify the highlighted part and select the appropriate material. "
            "This is a mechanical assembly with metal and plastic components."
        ],
    )

    # Prediction batching
    prediction_batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of prims per VLM call. 1 = one prim per call (default). "
        "N > 1 = group N prims into a single VLM call for faster throughput.",
    )


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[PipelineStep] = Field(
        min_length=1,
        description="Steps to re-run from cache",
    )

    # Can override user prompt for regeneration
    user_prompt: str | None = Field(
        default=None, description="Override user prompt for regeneration"
    )

    # Output format
    layer_only: bool = Field(
        default=False,
        description=(
            "Output only a material binding layer instead of a full USD. "
            "When true, preserves original scene structure."
        ),
    )
    coverage_policy: Literal["strict", "allow_partial"] | None = Field(
        default=None,
        description=(
            "Coverage qualification policy. strict fails closed when target prims "
            "lack usable predictions or bindings; allow_partial preserves a "
            "completed result with explicit partial readiness."
        ),
    )
    material_library: str | None = Field(
        default=None,
        description=(
            "Material library ID to predict and apply against. None keeps the "
            "library the session already resolved. Ignored when the session has "
            "custom uploaded materials."
        ),
        examples=["simready-category:Metal"],
    )
    material_profile: str | None = Field(
        default=None,
        description=(
            "Material authoring profile override: auto, display_color, "
            "preview_surface, openpbr_materialx, or omnipbr_mdl. None keeps the "
            "profile resolution the pipeline defaults to."
        ),
    )


class MaterialVariantSpec(BaseModel):
    """One material treatment to produce for an already-processed asset."""

    label: str = Field(
        min_length=1,
        max_length=120,
        description="Human-readable name shown next to the variant preview",
        examples=["Weathered steel"],
    )
    user_prompt: str | None = Field(
        default=None,
        description="Guidance handed to the VLM for this variant only",
        examples=["Treat every part as heavily weathered, rusted steel."],
    )
    material_library: str | None = Field(
        default=None,
        description="Material library ID for this variant (default: session library)",
        examples=["simready-category:Plastic"],
    )
    material_profile: str | None = Field(
        default=None,
        description="Material authoring profile for this variant",
        examples=["omnipbr_mdl"],
    )
    layer_only: bool = Field(
        default=False,
        description=(
            "Store this variant as a USD material binding layer over the "
            "original geometry instead of a full stage."
        ),
    )
    coverage_policy: Literal["strict", "allow_partial"] | None = Field(
        default=None,
        description="Coverage policy for this variant (default: session policy)",
    )
    steps: list[PipelineStep] | None = Field(
        default=None,
        description="Per-variant step override (default: the request-level steps)",
    )


class VariantsRequest(BaseModel):
    """Request to produce several material treatments of one session asset."""

    variants: list[MaterialVariantSpec] = Field(
        min_length=1,
        max_length=8,
        description="Variant specifications, executed serially in order",
    )
    steps: list[PipelineStep] = Field(
        default=[
            PipelineStep.BUILD_DATASET_PREPARE_DATASET,
            PipelineStep.PREDICT,
            PipelineStep.APPLY,
            PipelineStep.RENDER,
        ],
        description=(
            "Steps replayed from cache for every variant. The cached multi-view "
            "renders are reused; dataset preparation is included by default "
            "because that is the step a variant's user_prompt reaches."
        ),
    )
    reset: bool = Field(
        default=False,
        description="Discard previously stored variants before running",
    )
