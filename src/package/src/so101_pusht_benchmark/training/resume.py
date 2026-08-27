"""Read-only, fail-closed validation for production operator resume stages."""

from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Literal, cast

from ..evaluation.comparative_report import load_index_comparison_input
from .artifacts import ArtifactError, ArtifactIndex, sha256_file
from .identity import BundleIdentity
from .budgets import APPROVED_OPTIMIZER_UPDATES

ResumeStage = Literal["training", "bundle", "evaluation"]
_MODEL_IDS = {
    "dp_cnn": "dp-cnn-production",
    "dp_transformer": "dp-transformer-production",
    "ibc": "ibc-production",
    "lstm_gmm": "lstm-gmm-production",
}


def _exact_regular_tree(root: Path, expected: set[Path]) -> None:
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise ArtifactError(f"resume output is unavailable: {root}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ArtifactError("resume output must be a regular non-symlink directory")
    actual: set[Path] = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        item_mode = item.lstat().st_mode
        if stat.S_ISLNK(item_mode):
            raise ArtifactError(f"symlink resume artifact is forbidden: {relative}")
        if stat.S_ISREG(item_mode):
            actual.add(relative)
        elif not stat.S_ISDIR(item_mode):
            raise ArtifactError(f"non-regular resume artifact is forbidden: {relative}")
    if actual != expected:
        raise ArtifactError("resume output contains missing or unexpected files")


def _require_path(actual: Path, expected: Path, label: str) -> None:
    if actual != expected.resolve():
        raise ArtifactError(f"non-canonical resumed {label} path")


def validate_production_resume_artifact(
    index: ArtifactIndex,
    *,
    stage: ResumeStage,
    model: str,
    artifact_id: str,
    output: Path,
) -> dict[str, object]:
    """Authenticate one completed stage without changing index or artifact bytes."""
    if _MODEL_IDS.get(model) != artifact_id:
        raise ArtifactError("production artifact ID does not match expected model")
    root = output.absolute()
    artifact_root = index.artifact_root
    if root == artifact_root or artifact_root not in root.parents:
        raise ArtifactError("resume output is outside the canonical artifact root")
    contract = index.authenticate_stage(artifact_id, stage)
    record = index.record(artifact_id)
    identity = BundleIdentity.from_dict(record.get("identity"))
    if contract.get("identity") != identity.to_dict():
        raise ArtifactError("resumed producer stage identity mismatch")
    if (
        identity.model != model
        or identity.optimizer_updates != APPROVED_OPTIMIZER_UPDATES.get(model)
    ):
        raise ArtifactError("resumed production identity does not match model/update budget")
    if (
        record.get("deployment_scope") != "simulation_only"
        or record.get("training_eligible") is not True
    ):
        raise ArtifactError("resumed artifact is not production eligible")

    if stage == "training":
        if (
            record.get("result_status") != "full_training_complete"
            or record.get("comparison_eligible") is not False
        ):
            raise ArtifactError("resumed training status is not full_training_complete")
        checkpoint, config = index.require_trusted_production_checkpoint(artifact_id)
        receipt = index.verify(artifact_id, "production_receipt")
        training_log = index.verify(artifact_id, "training_log")
        _require_path(checkpoint, root / "checkpoints/latest.ckpt", "checkpoint")
        _require_path(config, root / "resolved_config.json", "training config")
        _require_path(receipt, root / "training_receipt.json", "training receipt")
        _require_path(training_log, root / "logs.json.txt", "training log")
        _exact_regular_tree(
            root,
            {
                Path("checkpoints/latest.ckpt"),
                Path("resolved_config.json"),
                Path("training_receipt.json"),
                Path("logs.json.txt"),
            },
        )
    elif stage == "bundle":
        if (
            contract.get("result_status") != "full_training_bundle_ready"
            or contract.get("bundle_schema") != 1
            or contract.get("comparison_eligible") is not False
        ):
            raise ArtifactError("resumed bundle producer contract mismatch")
        checkpoint = index.verify(artifact_id, "checkpoint")
        bundle = index.verify(artifact_id, "bundle")
        config = index.verify(artifact_id, "config")
        normalizer = index.verify(artifact_id, "normalizer")
        manifest = index.verify(artifact_id, "manifest")
        _require_path(bundle, root / "policy.safetensors", "bundle")
        _require_path(config, root / "resolved_config.json", "bundle config")
        _require_path(normalizer, root / "normalizer.json", "normalizer")
        _require_path(manifest, root / "bundle_manifest.json", "bundle manifest")
        try:
            manifest_value: object = json.loads(manifest.read_text(encoding="utf-8"))
            normalizer_value: object = json.loads(normalizer.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("malformed resumed bundle metadata") from exc
        if manifest_value != identity.bundle_manifest(checkpoint, config):
            raise ArtifactError("resumed bundle manifest identity mismatch")
        if not isinstance(normalizer_value, dict):
            raise ArtifactError("malformed resumed normalizer metadata")
        normalizer_document = cast("dict[str, object]", normalizer_value)
        if (
            normalizer_document.get("schema") != 1
            or normalizer_document.get("deployment_scope") != "simulation_only"
            or normalizer_document.get("training_eligible") is not True
            or normalizer_document.get("identity") != identity.to_dict()
            or normalizer_document.get("source_checkpoint_sha256") != sha256_file(checkpoint)
            or normalizer_document.get("resolved_config_sha256") != sha256_file(config)
            or not isinstance(normalizer_document.get("state"), dict)
        ):
            raise ArtifactError("resumed normalizer identity mismatch")
        _exact_regular_tree(
            root,
            {
                Path("policy.safetensors"),
                Path("resolved_config.json"),
                Path("normalizer.json"),
                Path("bundle_manifest.json"),
            },
        )
    else:
        if (
            contract.get("result_status") != "anchored_final_evaluation"
            or contract.get("metric_schema") != "pusht-so100-dxy-dyaw-v1"
            or contract.get("evaluation_seeds") != list(range(100000, 100100))
            or contract.get("step_cap") != 300
            or contract.get("fps") != 10
            or contract.get("comparison_eligible") is not True
        ):
            raise ArtifactError("resumed evaluation producer contract mismatch")
        comparison = load_index_comparison_input(index, artifact_id)
        if comparison.model != model:
            raise ArtifactError("resumed evaluation model identity mismatch")
        metrics = index.verify(artifact_id, "metrics")
        _require_path(metrics, root / "metrics.json", "evaluation metrics")
        try:
            metrics_value: object = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("malformed resumed evaluation metrics") from exc
        if not isinstance(metrics_value, dict):
            raise ArtifactError("malformed resumed evaluation metrics")
        rollouts = cast("dict[str, object]", metrics_value).get("rollouts")
        if not isinstance(rollouts, list):
            raise ArtifactError("malformed resumed evaluation rollouts")
        failures: list[dict[str, object]] = []
        for item in cast("list[object]", rollouts):
            if isinstance(item, dict):
                typed_item = cast("dict[str, object]", item)
                if typed_item.get("success") is not True:
                    failures.append(typed_item)
        traces = root / "failure_traces.json"
        expected_traces = (json.dumps(failures, indent=2, sort_keys=True) + "\n").encode()
        try:
            if traces.is_symlink() or traces.read_bytes() != expected_traces:
                raise ArtifactError("resumed evaluation failure traces mismatch")
        except OSError as exc:
            raise ArtifactError("resumed evaluation failure traces are unavailable") from exc
        _exact_regular_tree(root, {Path("metrics.json"), Path("failure_traces.json")})
    return {"status": "validated", "stage": stage, "model": model, "artifact_id": artifact_id}
