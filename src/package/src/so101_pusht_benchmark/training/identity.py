"""Digest-bound identities shared by checkpoints, bundles, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
from typing import cast

from ..integrations.paper_baselines.configs import PROFILES
from ..workspace import PACKAGE_ROOT
from .artifacts import ArtifactError, sha256_file
from .budgets import APPROVED_OPTIMIZER_UPDATES, LOCAL_OPTIMIZER_UPDATES

_STANFORD_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
_ROBOMIMIC_COMMIT = "62ed2de905caeb9133136e4d14d810a8b6baa96c"
_RUNTIME_LOCK = PACKAGE_ROOT / "environments/sim-runtime.lock"
_ENVIRONMENT_MANIFEST = PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json"


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ArtifactError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2_000_000:
        raise ArtifactError(f"{label} must be an integer in 1..2000000")
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    model: str
    policy_target: str
    policy_class: str
    policy_module: str
    workspace_target: str
    workspace_class: str
    workspace_module: str
    observation_steps: int
    horizon: int
    executed_actions: int
    optimizer_updates: int
    dataset_digest: str
    split_digest: str
    runtime_lock_digest: str
    environment_manifest_digest: str
    stanford_commit: str
    robomimic_commit: str

    def __post_init__(self) -> None:
        """Reject every field that differs from pinned runtime and model identities."""
        for label in (
            "model",
            "policy_target",
            "policy_class",
            "policy_module",
            "workspace_target",
            "workspace_class",
            "workspace_module",
            "stanford_commit",
            "robomimic_commit",
        ):
            _string(getattr(self, label), label)
        observation_steps = _positive_integer(self.observation_steps, "observation_steps")
        horizon = _positive_integer(self.horizon, "horizon")
        executed_actions = _positive_integer(self.executed_actions, "executed_actions")
        _positive_integer(self.optimizer_updates, "optimizer_updates")
        if observation_steps > horizon or executed_actions > horizon:
            raise ArtifactError("trusted horizon ranges are invalid")
        profile = PROFILES.get(self.model)
        if profile is None:
            raise ArtifactError(f"unknown model identity: {self.model}")
        expected_policy = f"{profile.policy_class.__module__}.{profile.policy_class.__name__}"
        workspace_module, workspace_class = profile.workspace_target.rsplit(".", 1)
        if (
            self.policy_target != expected_policy
            or self.policy_class != profile.policy_class.__name__
            or self.policy_module != profile.policy_class.__module__
            or self.workspace_target != profile.workspace_target
            or self.workspace_class != workspace_class
            or self.workspace_module != workspace_module
            or (self.observation_steps, self.horizon, self.executed_actions)
            != (profile.observation_steps, profile.horizon, profile.executed_actions)
            or self.optimizer_updates
            not in {1, profile.optimizer_updates, APPROVED_OPTIMIZER_UPDATES[self.model]}
            and not (
                os.environ.get("PUSHT_LOCAL_BUDGET") == "1"
                and self.optimizer_updates == LOCAL_OPTIMIZER_UPDATES[self.model]
            )
            or self.stanford_commit != _STANFORD_COMMIT
            or self.robomimic_commit != _ROBOMIMIC_COMMIT
        ):
            raise ArtifactError("trusted model identity does not match the pinned profile")
        for label in (
            "dataset_digest",
            "split_digest",
            "runtime_lock_digest",
            "environment_manifest_digest",
        ):
            _digest(getattr(self, label), label)
        if self.runtime_lock_digest != sha256_file(_RUNTIME_LOCK):
            raise ArtifactError("runtime lock identity mismatch")
        if self.environment_manifest_digest != sha256_file(_ENVIRONMENT_MANIFEST):
            raise ArtifactError("frozen environment manifest identity mismatch")

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", asdict(self))

    @classmethod
    def from_dict(cls, value: object) -> BundleIdentity:
        if not isinstance(value, dict):
            raise ArtifactError("bundle identity must be a mapping of SHA-256-bound fields")
        raw = cast("dict[str, object]", value)
        expected = set(cls.__dataclass_fields__)
        if set(raw) != expected:
            raise ArtifactError("bundle identity fields or SHA-256 bindings are incomplete")
        return cls(
            model=_string(raw["model"], "model"),
            policy_target=_string(raw["policy_target"], "policy_target"),
            policy_class=_string(raw["policy_class"], "policy_class"),
            policy_module=_string(raw["policy_module"], "policy_module"),
            workspace_target=_string(raw["workspace_target"], "workspace_target"),
            workspace_class=_string(raw["workspace_class"], "workspace_class"),
            workspace_module=_string(raw["workspace_module"], "workspace_module"),
            observation_steps=_positive_integer(raw["observation_steps"], "observation_steps"),
            horizon=_positive_integer(raw["horizon"], "horizon"),
            executed_actions=_positive_integer(raw["executed_actions"], "executed_actions"),
            optimizer_updates=_positive_integer(raw["optimizer_updates"], "optimizer_updates"),
            dataset_digest=_digest(raw["dataset_digest"], "dataset_digest"),
            split_digest=_digest(raw["split_digest"], "split_digest"),
            runtime_lock_digest=_digest(raw["runtime_lock_digest"], "runtime_lock_digest"),
            environment_manifest_digest=_digest(
                raw["environment_manifest_digest"], "environment_manifest_digest"
            ),
            stanford_commit=_string(raw["stanford_commit"], "stanford_commit"),
            robomimic_commit=_string(raw["robomimic_commit"], "robomimic_commit"),
        )

    def bundle_manifest(self, checkpoint: Path, config: Path) -> dict[str, object]:
        return {
            "schema": 1,
            "identity": self.to_dict(),
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "resolved_config_sha256": sha256_file(config),
        }


def trusted_identity(
    model: str,
    dataset_digest: str,
    split_digest: str,
    *,
    optimizer_updates: int = 1,
) -> BundleIdentity:
    profile = PROFILES.get(model)
    if profile is None:
        raise ArtifactError(f"unknown model identity: {model}")
    workspace_module, workspace_class = profile.workspace_target.rsplit(".", 1)
    return BundleIdentity(
        model=model,
        policy_target=f"{profile.policy_class.__module__}.{profile.policy_class.__name__}",
        policy_class=profile.policy_class.__name__,
        policy_module=profile.policy_class.__module__,
        workspace_target=profile.workspace_target,
        workspace_class=workspace_class,
        workspace_module=workspace_module,
        observation_steps=profile.observation_steps,
        horizon=profile.horizon,
        executed_actions=profile.executed_actions,
        optimizer_updates=optimizer_updates,
        dataset_digest=_digest(dataset_digest, "dataset_digest"),
        split_digest=_digest(split_digest, "split_digest"),
        runtime_lock_digest=sha256_file(_RUNTIME_LOCK),
        environment_manifest_digest=sha256_file(_ENVIRONMENT_MANIFEST),
        stanford_commit=_STANFORD_COMMIT,
        robomimic_commit=_ROBOMIMIC_COMMIT,
    )


def fixture_split_digest(dataset_digest: str) -> str:
    _digest(dataset_digest, "dataset_digest")
    return hashlib.sha256(f"ineligible-fixture:{dataset_digest}".encode()).hexdigest()
