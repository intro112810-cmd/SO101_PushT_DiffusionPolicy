"""Typed provenance contracts for deterministic sim-to-real replay.

Owns the frozen dataclasses, scalar constants, and exact array-shape validators
shared by history assembly, policy replay, and receipt serialization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

UInt8Image = NDArray[np.uint8]
Float32Vector = NDArray[np.float32]
PolicyLoad = Callable[[], tuple[object, object]]


class TensorToNumpy(Protocol):
    def detach(self) -> TensorToNumpy: ...
    def cpu(self) -> TensorToNumpy: ...
    def numpy(self, force: bool = False) -> NDArray[np.generic]: ...


FIXTURE_LINEAGE_ID = "fixture-local-dp_cnn-recovered-v3-seed0"
PRODUCTION_LINEAGE_ID = "local-dp_cnn-recovered-v4-seed0"
JOINT_EQUIVALENCE_DIGEST = "f0d1841fbd1e0846685a91cdfbf2f1ac7114b678aad1e8159d37e2f3e79f0c89"
PRODUCTION_JOINT_EQUIVALENCE_DIGEST = "8d60f850a7bad82ca57e5e5247414e8a8bcc8b1921c05eb56363dd0cc65e356e"
CAMERA_REGISTRATION_DIGEST = "f6453dcc3a48b66d7f7c0f01ea106934eddd196a312e045f93d0fcb0a500fdc3"
ARTIFACT_ROOT = Path("/home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark")
HISTORY_OBSERVATION_KEYS: Final = ("cam_top", "agent_pos")
EXECUTED_ACTIONS: Final = 8
FLOAT_TEXT_WIDTH = 17
RECEIPT_FIELDS: Final = (
    "schema",
    "mode",
    "policy",
    "policy_attempt",
    "artifact_id",
    "lineage_authority_digest",
    "lineage_digest",
    "joint_digest",
    "camera_digest",
    "sample_ids",
    "sample_digests",
    "camera_sha256s",
    "agent_pos_sha256s",
    "action_chunk_float32_2d",
    "seed",
    "latency_seconds",
    "deployment_valid",
    "hardware_actuation",
    "crop_randomizer_missing",
)


def validate_history_step_arrays(
    image: UInt8Image,
    agent_pos: Float32Vector,
    *,
    camera_sha256: str,
    agent_pos_sha256: str,
) -> None:
    """Validate the exact policy input contract for one history step."""
    if image.shape != (96, 96, 3) or image.dtype != np.uint8:
        raise RolloutViolation(
            RolloutCode.HISTORY_INCOMPLETE,
            "checkpoint image must be exact uint8[96,96,3]",
        )
    if agent_pos.shape != (5,) or agent_pos.dtype != np.float32:
        raise RolloutViolation(
            RolloutCode.HISTORY_INCOMPLETE,
            "agent_pos must be exact float32[5]",
        )
    if not bool(np.isfinite(agent_pos).all()):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "agent_pos must be finite")
    if hashlib.sha256(image.tobytes()).hexdigest() != camera_sha256:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "checkpoint image content")
    if hashlib.sha256(agent_pos.tobytes()).hexdigest() != agent_pos_sha256:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "agent_pos content")


def round_trip_float(value: float) -> float:
    """Force the same decimal spelling every process would re-read from JSON."""
    return float(f"{value:.{FLOAT_TEXT_WIDTH}g}")


@dataclass(frozen=True, slots=True)
class HistoryStep:
    """One provenance-bound observation with the exact policy input dtype."""

    sample_id: str
    sample_digest: str
    frame_digest: str
    camera_sha256: str
    agent_pos_sha256: str
    checkpoint_image: UInt8Image
    agent_pos: Float32Vector

    def __post_init__(self) -> None:
        """Reject any step that does not preserve the exact policy input contract."""
        validate_history_step_arrays(
            self.checkpoint_image,
            self.agent_pos,
            camera_sha256=self.camera_sha256,
            agent_pos_sha256=self.agent_pos_sha256,
        )


@dataclass(frozen=True, slots=True)
class HistoryEvidence:
    """Accepted receipt documents plus the authority digest they must satisfy."""

    samples: tuple[dict[str, object], ...]
    joint_document: dict[str, object]
    camera_document: dict[str, object]
    lineage_document: dict[str, object]
    lineage_authority_digest: str
    source_frame_path: Path


@dataclass(frozen=True, slots=True)
class PolicyRun:
    """Deterministic eight-action raw chunk with an injected latency value."""

    actions: Float32Vector
    latency_seconds: float
    policy: Literal["frozen", "fixture_deterministic_adapter"]

    def __post_init__(self) -> None:
        """Reject non-canonical eight-action chunks without clipping."""
        if self.actions.shape != (EXECUTED_ACTIONS, 2) or self.actions.dtype != np.float32:
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "action chunk shape/dtype")
        if not bool(np.isfinite(self.actions).all()):
            raise RolloutViolation(RolloutCode.R_NONFINITE, "action chunk must be finite")
        if bool(np.any(self.actions < -1.0)) or bool(np.any(self.actions > 1.0)):
            raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "action exceeds [-1,1]")


@dataclass(frozen=True, slots=True)
class InferenceReceipt:
    """Canonical, deterministic receipt emitted by the replay CLI."""

    schema: int
    mode: str
    policy: Literal["frozen", "fixture_deterministic_adapter"]
    policy_attempt: Literal["frozen", "fixture"]
    artifact_id: str
    lineage_authority_digest: str
    lineage_digest: str
    joint_digest: str
    camera_digest: str
    sample_ids: tuple[str, ...]
    sample_digests: tuple[str, ...]
    camera_sha256s: tuple[str, ...]
    agent_pos_sha256s: tuple[str, ...]
    action_chunk: Float32Vector
    seed: int
    latency_seconds: float
    deployment_valid: bool
    hardware_actuation: bool
    crop_randomizer_missing: bool

    @property
    def action_chunk_float32_2d(self) -> list[list[float]]:
        """Expose the stable eight-action chunk in the canonical wire shape."""
        return [[float(value) for value in row] for row in self.action_chunk.tolist()]

    def to_document(self) -> dict[str, object]:
        """Encode the receipt as the canonical machine-consumed mapping."""
        return {
            "schema": self.schema,
            "mode": self.mode,
            "policy": self.policy,
            "policy_attempt": self.policy_attempt,
            "artifact_id": self.artifact_id,
            "lineage_authority_digest": self.lineage_authority_digest,
            "lineage_digest": self.lineage_digest,
            "joint_digest": self.joint_digest,
            "camera_digest": self.camera_digest,
            "sample_ids": list(self.sample_ids),
            "sample_digests": list(self.sample_digests),
            "camera_sha256s": list(self.camera_sha256s),
            "agent_pos_sha256s": list(self.agent_pos_sha256s),
            "action_chunk_float32_2d": self.action_chunk_float32_2d,
            "seed": self.seed,
            "latency_seconds": round_trip_float(self.latency_seconds),
            "deployment_valid": self.deployment_valid,
            "hardware_actuation": self.hardware_actuation,
            "crop_randomizer_missing": self.crop_randomizer_missing,
        }
