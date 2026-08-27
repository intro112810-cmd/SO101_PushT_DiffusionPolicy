"""Trusted, non-executable inference metadata readers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

from ..integrations.paper_baselines.configs import PROFILES, SHAPE_META, workspace_config
from .artifacts import ArtifactError
from .identity import BundleIdentity


def _validate_key_tree(actual: object, template: object, label: str) -> None:
    if isinstance(template, dict):
        if not isinstance(actual, dict):
            raise ArtifactError(f"resolved config {label} must be a mapping")
        actual_mapping = cast("dict[str, object]", actual)
        template_mapping = cast("dict[str, object]", template)
        if set(actual_mapping) != set(template_mapping):
            raise ArtifactError(f"resolved config {label} keys mismatch")
        for key, expected in template_mapping.items():
            _validate_key_tree(actual_mapping[key], expected, f"{label}.{key}")
    elif isinstance(template, list) and not isinstance(actual, list):
        raise ArtifactError(f"resolved config {label} must be a list")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def read_trusted_config(path: Path, expected_model: str | None = None) -> dict[str, object]:
    """Read a resolved config only when its complete model identity is pinned."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("resolved config is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ArtifactError("resolved config must be an object")
    config = cast("dict[str, object]", raw)
    model = config.get("name")
    if not isinstance(model, str) or model not in PROFILES:
        raise ArtifactError("resolved config contains an unknown model identity")
    if expected_model is not None and model != expected_model:
        raise ArtifactError("resolved config model identity mismatch")
    profile = PROFILES[model]
    template = workspace_config(model, "/trusted/frozen-view", 0)
    _validate_key_tree(config, template, "root")
    policy = config.get("policy")
    expected_policy = f"{profile.policy_class.__module__}.{profile.policy_class.__name__}"
    if (
        config.get("_target_") != profile.workspace_target
        or config.get("horizon") != profile.horizon
        or config.get("n_obs_steps") != profile.observation_steps
        or config.get("n_action_steps") != profile.executed_actions
        or not isinstance(policy, dict)
        or cast("dict[str, object]", policy).get("_target_") != expected_policy
    ):
        raise ArtifactError("resolved config contains an unapproved model identity")
    if config.get("shape_meta") != SHAPE_META:
        raise ArtifactError("resolved config native shape identity mismatch")
    policy_values = cast("dict[str, object]", policy)
    if policy_values.get("shape_meta") != SHAPE_META:
        raise ArtifactError("resolved policy native shape identity mismatch")
    # JSON key sorting is storage-canonical, while upstream builds modality order
    # from insertion order. Restore the already-validated native namespace only.
    config["shape_meta"] = deepcopy(SHAPE_META)
    policy_values["shape_meta"] = deepcopy(SHAPE_META)
    return config


def read_normalizer_metadata(
    path: Path,
    expected_identity: BundleIdentity,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read normalizer schema only when every anchored identity field matches."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("normalizer metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ArtifactError("normalizer metadata must be an object")
    data = cast("dict[str, object]", raw)
    expected_root = {
        "schema",
        "deployment_scope",
        "training_eligible",
        "source_checkpoint_sha256",
        "resolved_config_sha256",
        "identity",
        "state",
    }
    if set(data) != expected_root:
        raise ArtifactError("normalizer metadata fields mismatch")
    state = data["state"]
    if (
        data["schema"] != 1
        or data["deployment_scope"] != "simulation_only"
        or data["training_eligible"] is not True
        or not isinstance(state, dict)
    ):
        raise ArtifactError("normalizer metadata schema is invalid")
    checkpoint_digest = _sha256(data["source_checkpoint_sha256"], "source checkpoint digest")
    config_digest = _sha256(data["resolved_config_sha256"], "resolved config digest")
    identity = BundleIdentity.from_dict(data["identity"])
    if identity != expected_identity:
        raise ArtifactError("normalizer metadata trusted identity mismatch")
    if checkpoint_digest != _sha256(
        expected_checkpoint_sha256, "expected source checkpoint digest"
    ):
        raise ArtifactError("normalizer metadata source checkpoint mismatch")
    if config_digest != _sha256(expected_config_sha256, "expected resolved config digest"):
        raise ArtifactError("normalizer metadata resolved config mismatch")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, value in cast("dict[str, object]", state).items():
        if not key.startswith("normalizer.") or not isinstance(value, dict):
            raise ArtifactError("normalizer metadata key is invalid")
        record = cast("dict[str, object]", value)
        if set(record) != {"shape", "dtype"}:
            raise ArtifactError("normalizer metadata tensor fields mismatch")
        shape, dtype = record["shape"], record["dtype"]
        if (
            not isinstance(shape, list)
            or not isinstance(dtype, str)
            or dtype not in {"torch.float32", "torch.float64"}
        ):
            raise ArtifactError("normalizer metadata tensor is invalid")
        raw_shape = cast("list[object]", shape)
        if not all(type(item) is int and item >= 0 for item in raw_shape):
            raise ArtifactError("normalizer metadata tensor is invalid")
        result[key] = (tuple(cast("list[int]", raw_shape)), dtype)
    if not result:
        raise ArtifactError("normalizer metadata has no state")
    return result
