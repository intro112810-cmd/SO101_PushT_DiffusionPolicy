"""Semantic validation for lineage-bound selectors and metadata documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .lineage_io import digest, object_list, object_mapping, read_json, relative_path
from .lineage_manifest import bundle_identity, validate_source_closure
from .lineage_types import DEFAULT_ROOTS, LineageError, LineageMember, LineageRoots, Scope

_FROZEN_BUNDLE_RUNTIME_LOCK_DIGEST = (
    "10776208a02c73299caf78249cecd8d1d6870e026ab456ec1c11087adc521b9a"
)


@dataclass(frozen=True, slots=True)
class DocumentInputs:
    """Verified paths and identities consumed by semantic document validation."""

    paths: dict[str, Path]
    digests: dict[str, str]
    identity: dict[str, object]
    members: tuple[LineageMember, ...]
    roots: LineageRoots
    artifact_id: str
    observed_sources: set[tuple[Scope, str]]


def _validate_artifact_index(
    path: Path,
    expected_identity: dict[str, object],
    digests: dict[str, str],
    artifact_id: str,
    members: tuple[LineageMember, ...],
) -> None:
    index = read_json(path, "artifact index")
    records = object_mapping(index.get("artifacts"), "artifact index records")
    record = object_mapping(records.get(artifact_id), "selected artifact record")
    bindings = {
        "bundle": "policy",
        "normalizer": "normalizer",
        "manifest": "bundle_manifest",
        "config": "resolved_config",
        "checkpoint": "source_checkpoint",
        "production_receipt": "training_receipt",
    }
    member_paths = {member.label: member.path for member in members}
    expected_paths = {field: member_paths[label] for field, label in bindings.items()}
    if record.get("identity") != expected_identity:
        raise LineageError("artifact index selected identity mismatch")
    for field, label in bindings.items():
        path_matches = record.get(f"{field}_path") == expected_paths[field]
        digest_matches = record.get(f"{field}_sha256") == digests[label]
        if not path_matches or not digest_matches:
            raise LineageError(f"artifact index selected {field} mismatch")
    if (
        record.get("deployment_scope") != "simulation_only"
        or record.get("training_eligible") is not True
        or record.get("result_status") != "anchored_final_evaluation"
    ):
        raise LineageError("artifact index selected authority state mismatch")


def _validate_bundle_documents(
    paths: dict[str, Path],
    digests: dict[str, str],
    expected: dict[str, object],
    training_expected: dict[str, object],
) -> None:
    bundle = read_json(paths["bundle_manifest"], "bundle manifest")
    normalizer = read_json(paths["normalizer"], "normalizer")
    receipt = read_json(paths["training_receipt"], "training receipt")
    if bundle.get("schema") != 1 or bundle.get("identity") != expected:
        raise LineageError("bundle manifest identity mismatch")
    if (
        bundle.get("source_checkpoint_sha256") != digests["source_checkpoint"]
        or bundle.get("resolved_config_sha256") != digests["resolved_config"]
    ):
        raise LineageError("bundle manifest member identity mismatch")
    if normalizer.get("schema") != 1 or normalizer.get("identity") != expected:
        raise LineageError("normalizer identity mismatch")
    if (
        normalizer.get("source_checkpoint_sha256") != digests["source_checkpoint"]
        or normalizer.get("resolved_config_sha256") != digests["resolved_config"]
    ):
        raise LineageError("normalizer member identity mismatch")
    if (
        receipt.get("schema") != "pusht-so100-full-training-v1"
        or receipt.get("completed") is not True
        or receipt.get("training_mode") != "full_production"
        or receipt.get("model") != "dp_cnn"
        or receipt.get("configured_optimizer_updates") != 400_000
        or receipt.get("executed_optimizer_updates") != 400_000
        or receipt.get("identity") != training_expected
    ):
        raise LineageError("source checkpoint training identity mismatch")


def _validate_config(config: dict[str, object]) -> None:
    shape = {
        "action": {"shape": [2]},
        "obs": {
            "agent_pos": {"shape": [5], "type": "low_dim"},
            "cam_top": {"shape": [3, 96, 96], "type": "rgb"},
        },
    }
    policy = object_mapping(config.get("policy"), "policy config")
    training = object_mapping(config.get("training"), "training config")
    task = object_mapping(config.get("task"), "task config")
    runner = object_mapping(task.get("env_runner"), "runner config")
    checks = (
        config.get("shape_meta") == shape,
        policy.get("shape_meta") == shape,
        config.get("horizon") == policy.get("horizon") == 16,
        config.get("n_obs_steps") == policy.get("n_obs_steps") == runner.get("n_obs_steps") == 2,
        config.get("n_action_steps")
        == policy.get("n_action_steps")
        == runner.get("n_action_steps")
        == 8,
        training.get("max_train_steps") == 400_000,
    )
    if not all(checks):
        raise LineageError("resolved config runtime identity mismatch")


def _validate_upstream(
    provenance: dict[str, object], members: tuple[LineageMember, ...], digests: dict[str, str]
) -> None:
    environment = object_mapping(provenance.get("environment"), "upstream environment")
    declared: list[tuple[str, str]] = []
    for raw in object_list(provenance.get("runtime_members"), "upstream runtime members"):
        item = object_mapping(raw, "upstream runtime member")
        path = relative_path(item.get("path"), "upstream runtime member")
        declared.append((path, digest(item.get("sha256"), path)))
    upstream = sorted(
        (member.path.removeprefix("05_references/external_repos/pushT-so100/"), member.sha256)
        for member in members
        if member.label.startswith("upstream_runtime_")
    )
    expected_environment = {
        "path": "environment.yml",
        "sha256": digests["upstream_environment"],
    }
    if (
        provenance.get("schema") != 1
        or sorted(declared) != upstream
        or environment != expected_environment
    ):
        raise LineageError("exact consumed upstream member inventory mismatch")


def validate_documents(inputs: DocumentInputs) -> None:
    """Validate selector, bundle metadata, config, and source inventories."""
    bundle_document_identity = bundle_identity(inputs.identity)
    training_identity = bundle_document_identity.copy()
    training_identity["runtime_lock_digest"] = _FROZEN_BUNDLE_RUNTIME_LOCK_DIGEST
    if inputs.roots == DEFAULT_ROOTS and inputs.artifact_id.endswith("v3-seed0"):
        bundle_document_identity = training_identity
    _validate_artifact_index(
        inputs.paths["artifact_index"],
        bundle_document_identity,
        inputs.digests,
        inputs.artifact_id,
        inputs.members,
    )
    _validate_bundle_documents(
        inputs.paths, inputs.digests, bundle_document_identity, training_identity
    )
    _validate_config(read_json(inputs.paths["resolved_config"], "resolved config"))
    provenance = read_json(
        inputs.paths["frozen_environment_manifest"], "frozen environment manifest"
    )
    _validate_upstream(provenance, inputs.members, inputs.digests)
    validate_source_closure(inputs.members, inputs.roots, inputs.observed_sources)
