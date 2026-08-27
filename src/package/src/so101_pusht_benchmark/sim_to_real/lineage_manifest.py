"""Strict lineage manifest and 400k identity parsing."""

from __future__ import annotations

from pathlib import Path

from .lineage_io import digest, object_list, object_mapping, read_json, relative_path
from .lineage_sources import source_inventory
from .lineage_types import (
    DEFAULT_ROOTS,
    LineageError,
    LineageMember,
    LineageRoots,
    MANIFEST_SCHEMA,
    Scope,
)

_CORE_LABELS = {
    "artifact_index",
    "policy",
    "normalizer",
    "bundle_manifest",
    "resolved_config",
    "source_checkpoint",
    "training_receipt",
    "runtime_lock",
    "frozen_environment_manifest",
    "upstream_environment",
}
_SOURCE_PREFIXES = ("route_source_", "stanford_source_", "robomimic_source_")
_AUTHORITY_FIELDS = {
    "artifact_id",
    "bundle_sha256",
    "camera_count",
    "camera_key",
    "camera_resolution",
    "dataset_digest",
    "decoded_actions",
    "frozen_environment_manifest_digest",
    "model",
    "observation_steps",
    "optimizer_updates",
    "policy_target",
    "prediction_horizon",
    "robomimic_commit",
    "runtime_lock_digest",
    "source_checkpoint_sha256",
    "split_digest",
    "stanford_commit",
    "workspace_target",
}


def parse_manifest(
    path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    tuple[LineageMember, ...],
    dict[str, object],
]:
    """Parse one exact manifest and reject duplicate members or paths."""
    document = read_json(path, "lineage authority manifest")
    if set(document) != {
        "schema",
        "artifact_id",
        "identity",
        "members",
        "runtime_fingerprint",
    }:
        raise LineageError("lineage authority manifest fields are incomplete")
    if document["schema"] != MANIFEST_SCHEMA or type(document["artifact_id"]) is not str:
        raise LineageError("lineage authority manifest schema or artifact identity is invalid")
    identity = object_mapping(document["identity"], "lineage identity")
    if set(identity) != _AUTHORITY_FIELDS:
        raise LineageError("lineage identity fields are incomplete")
    members: list[LineageMember] = []
    labels: set[str] = set()
    paths: set[tuple[str, str]] = set()
    for raw in object_list(document["members"], "lineage members"):
        item = object_mapping(raw, "lineage member")
        if set(item) != {"label", "scope", "path", "sha256"}:
            raise LineageError("lineage member record is malformed")
        label, raw_scope = item["label"], item["scope"]
        if type(label) is not str or not label:
            raise LineageError("lineage member label is malformed")
        if raw_scope == "artifact":
            scope: Scope = "artifact"
        elif raw_scope == "package":
            scope = "package"
        elif raw_scope == "project":
            scope = "project"
        elif raw_scope == "runtime":
            scope = "runtime"
        else:
            raise LineageError("lineage member scope is malformed")
        member_path = relative_path(item["path"], label)
        key = (scope, member_path)
        if label in labels or key in paths:
            raise LineageError("duplicate lineage member label or path")
        labels.add(label)
        paths.add(key)
        members.append(LineageMember(label, scope, member_path, digest(item["sha256"], label)))
    if not labels >= _CORE_LABELS:
        raise LineageError("lineage member inventory omits a required member")
    runtime_fingerprint = object_mapping(document["runtime_fingerprint"], "runtime fingerprint")
    return document, identity, tuple(members), runtime_fingerprint


def bundle_identity(identity: dict[str, object]) -> dict[str, object]:
    """Translate authority identity into the frozen bundle identity shape."""
    return {
        "model": identity["model"],
        "policy_target": identity["policy_target"],
        "policy_class": "DiffusionUnetHybridImagePolicy",
        "policy_module": "diffusion_policy.policy.diffusion_unet_hybrid_image_policy",
        "workspace_target": identity["workspace_target"],
        "workspace_class": "TrainDiffusionUnetHybridWorkspace",
        "workspace_module": "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace",
        "observation_steps": identity["observation_steps"],
        "horizon": identity["prediction_horizon"],
        "executed_actions": identity["decoded_actions"],
        "optimizer_updates": identity["optimizer_updates"],
        "dataset_digest": identity["dataset_digest"],
        "split_digest": identity["split_digest"],
        "runtime_lock_digest": identity["runtime_lock_digest"],
        "environment_manifest_digest": identity["frozen_environment_manifest_digest"],
        "stanford_commit": identity["stanford_commit"],
        "robomimic_commit": identity["robomimic_commit"],
    }


def validate_identity(identity: dict[str, object], artifact_id: str) -> None:
    """Require the exact supported 400k one-camera DP-CNN runtime identity."""
    exact = {
        "artifact_id": artifact_id,
        "camera_count": 1,
        "camera_key": "cam_top",
        "camera_resolution": [96, 96],
        "decoded_actions": 8,
        "model": "dp_cnn",
        "observation_steps": 2,
        "optimizer_updates": 400_000,
        "prediction_horizon": 16,
    }
    if any(
        identity.get(key) != value or type(identity.get(key)) is not type(value)
        for key, value in exact.items()
    ):
        raise LineageError("400k DP-CNN runtime identity mismatch")
    for field in (
        "bundle_sha256",
        "dataset_digest",
        "frozen_environment_manifest_digest",
        "runtime_lock_digest",
        "source_checkpoint_sha256",
        "split_digest",
    ):
        digest(identity[field], field)
    expected_policy = (
        "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy"
    )
    expected_workspace = (
        "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace."
        "TrainDiffusionUnetHybridWorkspace"
    )
    if (
        identity["policy_target"] != expected_policy
        or identity["workspace_target"] != expected_workspace
    ):
        raise LineageError("DP-CNN policy/workspace identity mismatch")
    if (
        identity["stanford_commit"] != "5ba07ac6661db573af695b419a7947ecb704690f"
        or identity["robomimic_commit"] != "62ed2de905caeb9133136e4d14d810a8b6baa96c"
    ):
        raise LineageError("frozen upstream commit identity mismatch")


def validate_source_closure(
    members: tuple[LineageMember, ...],
    roots: LineageRoots,
    observed: set[tuple[Scope, str]],
) -> None:
    """Require declared, AST-derived, and fresh-import source closures to agree."""
    declared = {
        (member.scope, member.path)
        for member in members
        if member.label.startswith(_SOURCE_PREFIXES)
    }
    expected = source_inventory(roots)
    observed_matches = roots != DEFAULT_ROOTS or declared == observed
    if declared != expected or not observed_matches:
        observed_required: set[tuple[Scope, str]] = observed if roots == DEFAULT_ROOTS else set()
        required = expected | observed_required
        missing = sorted(f"{scope}:{path}" for scope, path in required - declared)
        extra = sorted(f"{scope}:{path}" for scope, path in declared - required)
        raise LineageError(
            f"consumed Python source closure mismatch; missing={missing}, extra={extra}"
        )
