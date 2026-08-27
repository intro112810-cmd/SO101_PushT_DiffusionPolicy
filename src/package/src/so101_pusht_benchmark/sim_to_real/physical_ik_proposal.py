"""Canonical, immutable physical-IK proposal and collision-proof identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.physical_ik_collision import (
    CollisionSample,
    ObstacleTransform,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import BodyDegrees, SweptPath
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

_FLOAT_TEXT_WIDTH: Final = 17
_NUMERIC_KEYS: Final = frozenset({"fk_residual_m", "singularity_metric", "branch_delta_degrees"})


def round_trip_float(value: float) -> float:
    """Force one stable decimal spelling for every hash and JSON consumer."""
    return float(f"{value:.{_FLOAT_TEXT_WIDTH}g}")


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class PhysicalIKProposal:
    """Frozen body-only degree proposal with content-addressed collision evidence."""

    body_degrees: BodyDegrees
    fk_residual_m: float
    singularity_metric: float
    branch_delta_degrees: float
    swept_path: SweptPath
    clipping_performed: bool
    gripper_present: bool
    joint_equivalence_digest: str
    proposal_hash: str
    collision_samples: tuple[CollisionSample, ...] = ()
    model_digest: str = ""
    policy_digest: str = ""
    scene_pose_digest: str = ""
    obstacle_transforms: tuple[ObstacleTransform, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed, clipped, or unauthenticated collision evidence."""
        if len(self.body_degrees) != 5 or not all(
            math.isfinite(value) for value in self.body_degrees
        ):
            raise RolloutViolation(RolloutCode.R_NONFINITE, "body degrees must be finite")
        if not math.isfinite(self.fk_residual_m) or not math.isfinite(self.singularity_metric):
            raise RolloutViolation(RolloutCode.R_NONFINITE, "proposal evidence must be finite")
        if self.clipping_performed:
            raise RolloutViolation(
                RolloutCode.R_CLIPPING_REQUIRED, "clipped proposals are never promotable"
            )
        if self.gripper_present:
            raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "gripper must never be present")
        if (
            len(self.collision_samples) < 2
            or self.swept_path != tuple(sample.site_xyz for sample in self.collision_samples)
            or not all(sample.valid_digest() for sample in self.collision_samples)
        ):
            raise RolloutViolation(RolloutCode.R_COLLISION, "collision proof is missing or invalid")
        if not _sha256(self.model_digest) or not _sha256(self.policy_digest):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "model or policy digest is invalid")
        if (
            not _sha256(self.scene_pose_digest)
            or len(self.obstacle_transforms) != 2
            or any(
                sample.pose_digest != self.scene_pose_digest for sample in self.collision_samples
            )
            or any(
                sample.obstacle_transforms != self.obstacle_transforms
                for sample in self.collision_samples
            )
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "scene pose binding is invalid")

    def to_document(self) -> dict[str, object]:
        return {
            "schema": 1,
            "mode": "physical_body_only_ik_proposal",
            "joint_order": list(JOINT_ORDER),
            "body_degrees": list(self.body_degrees),
            "fk_residual_m": round_trip_float(self.fk_residual_m),
            "singularity_metric": round_trip_float(self.singularity_metric),
            "branch_delta_degrees": round_trip_float(self.branch_delta_degrees),
            "swept_path": [list(point) for point in self.swept_path],
            "collision_samples": [sample.to_document() for sample in self.collision_samples],
            "model_digest": self.model_digest,
            "policy_digest": self.policy_digest,
            "scene_pose_digest": self.scene_pose_digest,
            "obstacle_transforms": [
                [name, list(transform)] for name, transform in self.obstacle_transforms
            ],
            "clipping_performed": self.clipping_performed,
            "gripper_present": self.gripper_present,
            "joint_equivalence_digest": self.joint_equivalence_digest,
            "proposal_hash": self.proposal_hash,
        }


def physical_ik_proposal_hash(document: dict[str, object]) -> str:
    payload: dict[str, object] = {}
    for key, value in document.items():
        if key in _NUMERIC_KEYS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RolloutViolation(RolloutCode.R_NONFINITE, f"{key} must be numeric")
            payload[key] = round_trip_float(float(value))
        else:
            payload[key] = value
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
