"""Deterministic, digest-locked aggregation of four final model evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
from collections.abc import Callable
from typing import cast

import numpy as np

from ..training.artifacts import ArtifactError, ArtifactIndex
from ..training.identity import BundleIdentity
from ..workspace import (
    WorkspacePolicyError,
    runtime_artifact_root,
    validate_report_path,
)

MODEL_ORDER = ("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")
_EVALUATION_SEEDS = tuple(range(100000, 100100))
_METRIC_SCHEMA = "pusht-so100-dxy-dyaw-v1"
_ENVIRONMENT_MANIFEST = "configs/provenance/pusht_so100_upstream.json"
_RUNTIME_LOCK = "environments/sim-runtime.lock"
_REPORT_SCHEMA = "pusht-so100-four-model-comparison-v1"
_INPUT_ROOT_FIELDS = {
    "schema",
    "artifact_id",
    "environment_manifest",
    "runtime_lock",
    "artifact_record",
    "metrics",
}
_RECORD_FIELDS = {
    "deployment_scope",
    "training_eligible",
    "comparison_eligible",
    "result_status",
    "identity",
    "metrics_sha256",
}
_METRIC_FIELDS = {
    "schema",
    "metric_schema",
    "model",
    "identity",
    "deployment_scope",
    "training_eligible",
    "evaluation_seeds",
    "step_cap",
    "fps",
    "observation_steps",
    "horizon",
    "executed_actions",
    "optimizer_updates",
    "wall_time_s",
    "eval/success_rate",
    "eval/mean_dxy",
    "eval/mean_dyaw",
    "eval/mean_duration_s",
    "rollouts",
}
_ROLLOUT_FIELDS = {
    "seed",
    "policy_seed",
    "success",
    "dxy",
    "dyaw",
    "duration_s",
    "steps",
    "terminated",
    "truncated",
}


class ComparisonError(ValueError):
    """Raised when final evaluations cannot be compared without provenance mixing."""


@dataclass(frozen=True, slots=True)
class ComparisonInput:
    artifact_id: str
    model: str
    identity: BundleIdentity
    wall_time_s: float
    success_rate: float
    mean_dxy: float
    mean_dyaw: float
    mean_duration_s: float
    rollout_aggregates: dict[str, int | float]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must be an object")
    return cast("dict[str, object]", value)


def _exact_fields(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ComparisonError(f"{label} fields mismatch")


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ComparisonError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ComparisonError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ComparisonError(f"{label} must be a finite number")
    result = float(cast("int | float", value))
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ComparisonError(f"{label} must be a finite number >= {minimum}")
    return result


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _parse_rollouts(value: object) -> tuple[list[dict[str, object]], dict[str, int | float]]:
    if not isinstance(value, list):
        raise ComparisonError("rollouts must contain exactly one result per evaluation seed")
    raw_rollouts = cast("list[object]", value)
    if len(raw_rollouts) != len(_EVALUATION_SEEDS):
        raise ComparisonError("rollouts must contain exactly one result per evaluation seed")
    rollouts: list[dict[str, object]] = []
    for expected_seed, item in zip(_EVALUATION_SEEDS, raw_rollouts, strict=True):
        rollout = _mapping(item, "rollout")
        _exact_fields(rollout, _ROLLOUT_FIELDS, "rollout")
        if _integer(rollout["seed"], "rollout seed") != expected_seed:
            raise ComparisonError("rollout seed order must be exactly 100000..100099")
        _integer(rollout["policy_seed"], "policy seed")
        if type(rollout["success"]) is not bool:
            raise ComparisonError("rollout success must be bool")
        if type(rollout["terminated"]) is not bool or type(rollout["truncated"]) is not bool:
            raise ComparisonError("rollout termination fields must be bool")
        success = rollout["success"]
        terminated = rollout["terminated"]
        truncated = rollout["truncated"]
        if success != terminated or (terminated and truncated):
            raise ComparisonError("rollout success/termination semantics mismatch")
        steps = _integer(rollout["steps"], "rollout steps", minimum=1)
        if steps > 300:
            raise ComparisonError("rollout exceeds the exact 300 step cap")
        duration = _finite(rollout["duration_s"], "rollout duration", minimum=0.0)
        if duration != steps / 10:
            raise ComparisonError("rollout duration does not match FPS 10")
        _finite(rollout["dxy"], "rollout dxy")
        _finite(rollout["dyaw"], "rollout dyaw")
        rollouts.append(rollout)
    steps = [cast("int", item["steps"]) for item in rollouts]
    return rollouts, {
        "count": len(rollouts),
        "successes": sum(cast("bool", item["success"]) for item in rollouts),
        "failures": sum(not cast("bool", item["success"]) for item in rollouts),
        "terminated": sum(cast("bool", item["terminated"]) for item in rollouts),
        "truncated": sum(cast("bool", item["truncated"]) for item in rollouts),
        "total_steps": sum(steps),
        "mean_steps": float(np.mean(steps)),
        "min_steps": min(steps),
        "max_steps": max(steps),
    }


def _require_aggregate(metrics: dict[str, object], key: str, values: list[float]) -> float:
    expected = float(np.mean(values))
    actual = _finite(metrics[key], key)
    if actual != expected:
        raise ComparisonError(f"{key} does not match required rollout aggregate")
    return actual


def _parse_document(document: dict[str, object]) -> ComparisonInput:
    _exact_fields(document, _INPUT_ROOT_FIELDS, "evaluation result")
    if document["schema"] != 1:
        raise ComparisonError("evaluation result schema must be 1")
    artifact_id = _string(document["artifact_id"], "artifact id")
    if document["environment_manifest"] != _ENVIRONMENT_MANIFEST:
        raise ComparisonError("frozen environment manifest path mismatch")
    if document["runtime_lock"] != _RUNTIME_LOCK:
        raise ComparisonError("frozen runtime lock path mismatch")

    record = _mapping(document["artifact_record"], "artifact record")
    _exact_fields(record, _RECORD_FIELDS, "artifact record")
    if (
        record["deployment_scope"] != "simulation_only"
        or record["training_eligible"] is not True
        or record["comparison_eligible"] is not True
        or record["result_status"] != "anchored_final_evaluation"
    ):
        raise ComparisonError(
            "comparison requires result_status anchored_final_evaluation and comparison_eligible"
        )

    metrics = _mapping(document["metrics"], "evaluation metrics")
    _exact_fields(metrics, _METRIC_FIELDS, "evaluation metrics")
    expected_digest = _digest(record["metrics_sha256"], "anchored metrics digest")
    if _sha256(_canonical_bytes(metrics)) != expected_digest:
        raise ComparisonError("metrics digest mismatch")
    if record["identity"] != metrics["identity"]:
        raise ComparisonError("artifact and metrics identities differ")
    try:
        identity = BundleIdentity.from_dict(record["identity"])
    except ArtifactError as exc:
        raise ComparisonError(str(exc)) from exc
    model = _string(metrics["model"], "metrics model")
    if model != identity.model:
        raise ComparisonError("metrics model and trusted artifact identity differ")
    if metrics["schema"] != 1 or metrics["metric_schema"] != _METRIC_SCHEMA:
        raise ComparisonError("metric schema/version must be pusht-so100-dxy-dyaw-v1 schema 1")
    if metrics["deployment_scope"] != "simulation_only" or metrics["training_eligible"] is not True:
        raise ComparisonError("only production simulation evaluations are comparison eligible")
    seeds = metrics["evaluation_seeds"]
    if not isinstance(seeds, list):
        raise ComparisonError("evaluation seeds must be exactly ordered 100000..100099")
    seed_values = cast("list[object]", seeds)
    if (
        any(type(seed) is not int for seed in seed_values)
        or tuple(seed_values) != _EVALUATION_SEEDS
    ):
        raise ComparisonError("evaluation seeds must be exactly ordered 100000..100099")
    if metrics["step_cap"] != 300 or type(metrics["step_cap"]) is not int:
        raise ComparisonError("evaluation step cap must be exactly 300")
    if metrics["fps"] != 10 or type(metrics["fps"]) is not int:
        raise ComparisonError("evaluation FPS must be exactly 10")
    if (
        metrics["observation_steps"] != identity.observation_steps
        or metrics["horizon"] != identity.horizon
        or metrics["executed_actions"] != identity.executed_actions
    ):
        raise ComparisonError("model horizon identity mismatch")
    if metrics["optimizer_updates"] != identity.optimizer_updates:
        raise ComparisonError("optimizer update identity mismatch")
    from ..training.budgets import APPROVED_OPTIMIZER_UPDATES

    if identity.optimizer_updates != APPROVED_OPTIMIZER_UPDATES.get(model):
        raise ComparisonError("comparison requires the approved full-production update budget")

    rollouts, rollout_aggregates = _parse_rollouts(metrics["rollouts"])
    success_rate = _require_aggregate(
        metrics,
        "eval/success_rate",
        [float(cast("bool", item["success"])) for item in rollouts],
    )
    mean_dxy = _require_aggregate(
        metrics, "eval/mean_dxy", [cast("float", item["dxy"]) for item in rollouts]
    )
    mean_dyaw = _require_aggregate(
        metrics, "eval/mean_dyaw", [cast("float", item["dyaw"]) for item in rollouts]
    )
    mean_duration = _require_aggregate(
        metrics,
        "eval/mean_duration_s",
        [cast("float", item["duration_s"]) for item in rollouts],
    )
    return ComparisonInput(
        artifact_id=artifact_id,
        model=model,
        identity=identity,
        wall_time_s=_finite(metrics["wall_time_s"], "wall time", minimum=0.0),
        success_rate=success_rate,
        mean_dxy=mean_dxy,
        mean_dyaw=mean_dyaw,
        mean_duration_s=mean_duration,
        rollout_aggregates=rollout_aggregates,
    )


def _input_documents(directory: Path) -> list[dict[str, object]]:
    try:
        mode = directory.lstat().st_mode
    except OSError as exc:
        raise ComparisonError(f"evaluation directory is unavailable: {directory}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ComparisonError("evaluation directory must be a regular non-symlink directory")
    entries = sorted(directory.iterdir(), key=lambda item: item.name)
    if not entries or any(
        item.suffix != ".json" or item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ComparisonError("evaluation directory must contain JSON evaluation results only")
    documents: list[dict[str, object]] = []
    for path in entries:
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ComparisonError(f"malformed evaluation result: {path.name}") from exc
        documents.append(_mapping(value, f"evaluation result {path.name}"))
    return documents


def _comparison_inputs(documents: list[dict[str, object]]) -> tuple[ComparisonInput, ...]:
    discovered: list[str] = []
    artifact_ids: list[str] = []
    for document in documents:
        metrics = _mapping(document.get("metrics"), "evaluation metrics")
        model = metrics.get("model")
        discovered.append(model if isinstance(model, str) else "<invalid>")
        artifact = document.get("artifact_id")
        artifact_ids.append(artifact if isinstance(artifact, str) else "<invalid>")
    if len(documents) != len(MODEL_ORDER) or sorted(discovered) != sorted(MODEL_ORDER):
        raise ComparisonError(
            "unique complete model set must be dp_cnn, dp_transformer, ibc, lstm_gmm"
        )
    if len(set(artifact_ids)) != len(MODEL_ORDER):
        raise ComparisonError("anchored artifact identities must be unique")
    parsed_inputs = [_parse_document(document) for document in documents]
    parsed = {item.model: item for item in parsed_inputs}
    ordered = tuple(parsed[model] for model in MODEL_ORDER)
    first = ordered[0].identity
    for item in ordered[1:]:
        if item.identity.dataset_digest != first.dataset_digest:
            raise ComparisonError("dataset digest mismatch across final evaluations")
        if item.identity.split_digest != first.split_digest:
            raise ComparisonError("split digest mismatch across final evaluations")
        if item.identity.runtime_lock_digest != first.runtime_lock_digest:
            raise ComparisonError("runtime lock digest mismatch across final evaluations")
        if item.identity.environment_manifest_digest != first.environment_manifest_digest:
            raise ComparisonError("environment manifest digest mismatch across final evaluations")
    return ordered


def load_comparison_inputs(directory: Path) -> tuple[ComparisonInput, ...]:
    """Load exactly one trusted final evaluation per model in canonical model order."""
    return _comparison_inputs(_input_documents(directory))


def load_index_comparison_input(index: ArtifactIndex, artifact_id: str) -> ComparisonInput:
    """Validate one authenticated evaluator-anchored final result."""
    contract = index.authenticate_stage(artifact_id, "evaluation")
    if (
        contract.get("result_status") != "anchored_final_evaluation"
        or contract.get("metric_schema") != _METRIC_SCHEMA
        or contract.get("evaluation_seeds") != list(_EVALUATION_SEEDS)
        or contract.get("step_cap") != 300
        or contract.get("fps") != 10
        or contract.get("comparison_eligible") is not True
    ):
        raise ComparisonError("authenticated final evaluation contract mismatch")
    record = index.record(artifact_id)
    metrics_path = index.verify(artifact_id, "metrics")
    try:
        value: object = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"malformed anchored evaluation: {artifact_id}") from exc
    metrics = _mapping(value, f"anchored evaluation {artifact_id}")
    return _parse_document(
        {
            "schema": 1,
            "artifact_id": artifact_id,
            "environment_manifest": _ENVIRONMENT_MANIFEST,
            "runtime_lock": _RUNTIME_LOCK,
            "artifact_record": {
                "deployment_scope": record.get("deployment_scope"),
                "training_eligible": record.get("training_eligible"),
                "comparison_eligible": record.get("comparison_eligible"),
                "result_status": record.get("result_status"),
                "identity": record.get("identity"),
                "metrics_sha256": _sha256(_canonical_bytes(metrics)),
            },
            "metrics": metrics,
        }
    )


def load_index_comparison_inputs(
    index: ArtifactIndex, artifact_ids: tuple[str, ...]
) -> tuple[ComparisonInput, ...]:
    """Build comparison inputs from four evaluator-anchored artifact records."""
    if len(artifact_ids) != len(MODEL_ORDER) or len(set(artifact_ids)) != len(MODEL_ORDER):
        raise ComparisonError("exactly four unique final evaluation artifact IDs are required")
    inputs = tuple(load_index_comparison_input(index, item) for item in artifact_ids)
    if sorted(item.model for item in inputs) != sorted(MODEL_ORDER):
        raise ComparisonError(
            "unique complete model set must be dp_cnn, dp_transformer, ibc, lstm_gmm"
        )
    if len({item.artifact_id for item in inputs}) != len(MODEL_ORDER):
        raise ComparisonError("anchored artifact identities must be unique")
    parsed = {item.model: item for item in inputs}
    ordered = tuple(parsed[model] for model in MODEL_ORDER)
    first = ordered[0].identity
    for item in ordered[1:]:
        if item.identity.dataset_digest != first.dataset_digest:
            raise ComparisonError("dataset digest mismatch across final evaluations")
        if item.identity.split_digest != first.split_digest:
            raise ComparisonError("split digest mismatch across final evaluations")
        if item.identity.runtime_lock_digest != first.runtime_lock_digest:
            raise ComparisonError("runtime lock digest mismatch across final evaluations")
        if item.identity.environment_manifest_digest != first.environment_manifest_digest:
            raise ComparisonError("environment manifest digest mismatch across final evaluations")
    return ordered


def _report(inputs: tuple[ComparisonInput, ...]) -> dict[str, object]:
    identity = inputs[0].identity
    return {
        "schema": _REPORT_SCHEMA,
        "provenance": {
            "dataset_digest": identity.dataset_digest,
            "split_digest": identity.split_digest,
            "environment_manifest": _ENVIRONMENT_MANIFEST,
            "environment_manifest_digest": identity.environment_manifest_digest,
            "runtime_lock": _RUNTIME_LOCK,
            "runtime_lock_digest": identity.runtime_lock_digest,
            "evaluation_seeds": list(_EVALUATION_SEEDS),
            "step_cap": 300,
            "fps": 10,
            "metric_schema": _METRIC_SCHEMA,
            "result_status": "anchored_final_evaluation",
            "comparison_eligible": True,
        },
        "models": [
            {
                "artifact_id": item.artifact_id,
                "model": item.model,
                "horizons": {
                    "observation_steps": item.identity.observation_steps,
                    "horizon": item.identity.horizon,
                    "executed_actions": item.identity.executed_actions,
                },
                "optimizer_updates": item.identity.optimizer_updates,
                "wall_time_s": item.wall_time_s,
                "success_rate": item.success_rate,
                "mean_dxy": item.mean_dxy,
                "mean_dyaw": item.mean_dyaw,
                "mean_duration_s": item.mean_duration_s,
                "rollout_aggregates": item.rollout_aggregates,
            }
            for item in inputs
        ],
    }


def _number(value: object) -> str:
    if type(value) is int:
        return str(value)
    return format(cast("float", value), ".17g")


def _markdown(report: dict[str, object]) -> str:
    provenance = cast("dict[str, object]", report["provenance"])
    rows = cast("list[dict[str, object]]", report["models"])
    lines = [
        "# PushT SO-100 Four-Model Comparison",
        "",
        f"- Dataset digest: `{provenance['dataset_digest']}`",
        f"- Split digest: `{provenance['split_digest']}`",
        f"- Frozen environment: `{provenance['environment_manifest']}` (`{provenance['environment_manifest_digest']}`)",
        f"- Runtime lock: `{provenance['runtime_lock']}` (`{provenance['runtime_lock_digest']}`)",
        "- Evaluation: seeds `100000..100099` in order, 300-step cap, FPS 10",
        f"- Metric schema: `{provenance['metric_schema']}`",
        "- Eligibility: `anchored_final_evaluation`, `comparison_eligible=true`",
        "",
        "| Model | Obs | Horizon | Actions | Updates | Wall time (s) | Success rate | Mean dxy | Mean dyaw | Mean duration (s) | Rollouts | Successes | Total steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        horizons = cast("dict[str, int]", row["horizons"])
        aggregates = cast("dict[str, int | float]", row["rollout_aggregates"])
        lines.append(
            "| "
            + " | ".join(
                (
                    cast("str", row["model"]),
                    str(horizons["observation_steps"]),
                    str(horizons["horizon"]),
                    str(horizons["executed_actions"]),
                    str(row["optimizer_updates"]),
                    _number(row["wall_time_s"]),
                    _number(row["success_rate"]),
                    _number(row["mean_dxy"]),
                    _number(row["mean_dyaw"]),
                    _number(row["mean_duration_s"]),
                    str(aggregates["count"]),
                    str(aggregates["successes"]),
                    str(aggregates["total_steps"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _canonical_output(output_dir: Path) -> Path:
    reports_root = (runtime_artifact_root() / "reports").resolve()
    try:
        root_mode = reports_root.lstat().st_mode
    except OSError as exc:
        raise ComparisonError("reports root is unavailable") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ComparisonError("reports root must be a regular directory")
    lexical = output_dir.expanduser().absolute()
    if lexical.parent.resolve() != reports_root or lexical.name in {"", ".", ".."}:
        raise ComparisonError("comparison output must be a direct child of the reports root")
    return lexical


def _validated_output(output_dir: Path) -> tuple[Path, Path]:
    lexical = _canonical_output(output_dir)
    if lexical.exists() or lexical.is_symlink():
        raise ComparisonError(f"comparison output already exists: {lexical}")
    staging = lexical.with_name(f".{lexical.name}.tmp")
    if staging.exists() or staging.is_symlink():
        raise ComparisonError(f"comparison staging already exists: {staging}")
    return lexical, staging


def _report_bytes(inputs: tuple[ComparisonInput, ...]) -> tuple[bytes, bytes]:
    report = _report(inputs)
    return (
        (json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(),
        _markdown(report).encode(),
    )


def _write_report(
    inputs: tuple[ComparisonInput, ...],
    output_dir: Path,
    after_json_write: Callable[[Path], None] | None = None,
) -> tuple[Path, Path]:
    output, staging = _validated_output(output_dir)
    json_bytes, markdown_bytes = _report_bytes(inputs)
    staging_created = False
    published = False
    try:
        staging.mkdir()
        staging_created = True
        json_path = validate_report_path(staging / "comparison.json")
        markdown_path = validate_report_path(staging / "comparison.md")
        json_path.write_bytes(json_bytes)
        if after_json_write is not None:
            after_json_write(json_path)
        markdown_path.write_bytes(markdown_bytes)
        staging.replace(output)
        published = True
        final_json = validate_report_path(output / "comparison.json")
        final_markdown = validate_report_path(output / "comparison.md")
    except (WorkspacePolicyError, OSError) as exc:
        if published:
            shutil.rmtree(output, ignore_errors=True)
        elif staging_created:
            shutil.rmtree(staging, ignore_errors=True)
        raise ComparisonError(str(exc)) from exc
    except BaseException:
        if published:
            shutil.rmtree(output, ignore_errors=True)
        elif staging_created:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_json, final_markdown


def write_comparative_report(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    after_json_write: Callable[[Path], None] | None = None,
) -> tuple[Path, Path]:
    """Validate comparison-ready inputs, then atomically publish both reports."""
    return _write_report(
        load_comparison_inputs(evaluation_dir), output_dir, after_json_write=after_json_write
    )


def write_comparative_report_from_index(
    index: ArtifactIndex, artifact_ids: tuple[str, ...], output_dir: Path
) -> tuple[Path, Path]:
    """Publish a report directly from four evaluator-anchored artifact records."""
    return _write_report(load_index_comparison_inputs(index, artifact_ids), output_dir)


def _read_existing_report_files(output: Path) -> tuple[bytes, bytes]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(output, directory_flags | nofollow)
    except OSError as exc:
        raise ComparisonError(
            "existing comparison report must be a regular non-symlink directory"
        ) from exc
    try:
        # The fd-relative listing is required to avoid a path re-resolution race.
        if set(os.listdir(directory_fd)) != {  # noqa: PTH208
            "comparison.json",
            "comparison.md",
        }:
            raise ComparisonError(
                "existing comparison report must contain exactly comparison.json and comparison.md"
            )
        contents: list[bytes] = []
        for name in ("comparison.json", "comparison.md"):
            try:
                file_fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | nofollow, dir_fd=directory_fd)
            except OSError as exc:
                raise ComparisonError(
                    "existing comparison report files must be regular and non-symlinked"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise ComparisonError(
                        "existing comparison report files must be regular and non-symlinked"
                    )
                chunks: list[bytes] = []
                while chunk := os.read(file_fd, 1024 * 1024):
                    chunks.append(chunk)
                contents.append(b"".join(chunks))
            finally:
                os.close(file_fd)
    finally:
        os.close(directory_fd)
    return contents[0], contents[1]


def validate_existing_comparative_report_from_index(
    index: ArtifactIndex, artifact_ids: tuple[str, ...], output_dir: Path
) -> tuple[Path, Path]:
    """Validate an existing report byte-for-byte against current anchored inputs."""
    output = _canonical_output(output_dir)
    expected = _report_bytes(load_index_comparison_inputs(index, artifact_ids))
    actual = _read_existing_report_files(output)
    if actual != expected:
        raise ComparisonError(
            "existing comparison report bytes do not match the current anchored evaluations"
        )
    return output / "comparison.json", output / "comparison.md"
