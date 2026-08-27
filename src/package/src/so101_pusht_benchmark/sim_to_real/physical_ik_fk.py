"""Shared primitives for the owned no-clipping physical IK planner.

Owns joint mapping (affine degree<->mapped radian, never clipped), the pinned
joint-equivalence receipt digest, the Cartesian target contract, and the
MuJoCo site/jacobian primitives used by the DLS solver and FK verifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim.scene import Scene
from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.policy_types import (
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.replay_types import (
    JOINT_EQUIVALENCE_DIGEST,
    PRODUCTION_JOINT_EQUIVALENCE_DIGEST,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt

BodyDegrees = tuple[float, float, float, float, float]
BodyRadians = tuple[float, float, float, float, float]
SweptPoint = tuple[float, float, float]
SweptPath = tuple[SweptPoint, ...]

Float64Array = NDArray[np.float64]
_SHA_HEX = frozenset("0123456789abcdef")
FREE_JOINTS: Final = 5


def _hex_digest(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA_HEX for character in value.lower())


def validate_joint_equivalence_digest(digest: str) -> None:
    """Reject any joint-equivalence receipt that is not the Todo 8 pin."""
    if not _hex_digest(digest):
        raise RolloutViolation(
            RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint equivalence digest must be SHA-256"
        )
    if digest.lower() != digest or digest not in {
        JOINT_EQUIVALENCE_DIGEST, PRODUCTION_JOINT_EQUIVALENCE_DIGEST
    }:
        raise RolloutViolation(
            RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint equivalence receipt has drifted"
        )


@dataclass(frozen=True, slots=True)
class JointDomain:
    """Approved physical-degree and mapped-radian bounds for one joint."""

    physical: tuple[float, float]
    mapped: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject non-finite or inverted bounds at construction."""
        if not (
            self.physical[0] < self.physical[1]
            and self.mapped[0] < self.mapped[1]
            and all(math.isfinite(value) for value in (*self.physical, *self.mapped))
        ):
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "joint domain is invalid")


def _domain(
    joint: str, policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy
) -> JointDomain:
    order = policy.joint_domains.joint_order
    physical = policy.joint_domains.physical_degrees
    mapped = policy.joint_domains.mapped_radians
    try:
        index = order.index(joint)
    except ValueError as exc:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, f"{joint} is missing from the joint order"
        ) from exc
    return JointDomain(
        (physical[index].minimum, physical[index].maximum),
        (mapped[index].minimum, mapped[index].maximum),
    )


def build_joint_domains(
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> dict[str, JointDomain]:
    """Bind approved physical/mapped bounds in canonical joint order."""
    if tuple(policy.joint_domains.joint_order) != JOINT_ORDER:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "joint order drift")
    domains = {joint: _domain(joint, policy) for joint in JOINT_ORDER}
    if len(domains) != 5:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "joint domains are incomplete")
    return domains


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be finite")
    return number


def parse_body_degrees(value: object) -> BodyDegrees:
    """Parse a measured body-degree seed; never accept a gripper key."""
    if isinstance(value, Mapping):
        if "gripper" in value:
            raise RolloutViolation(
                RolloutCode.R_OUT_OF_RANGE, "gripper must never appear in a body proposal"
            )
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "body degrees must be a sequence")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "body degrees must be five values")
    if len(cast("Sequence[object]", value)) != 5:
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "body degrees must be five values")
    sequence = cast("Sequence[object]", value)
    return (
        _finite_number(sequence[0], "body degree"),
        _finite_number(sequence[1], "body degree"),
        _finite_number(sequence[2], "body degree"),
        _finite_number(sequence[3], "body degree"),
        _finite_number(sequence[4], "body degree"),
    )


def parse_cartesian_target(target: CartesianProposalReceipt) -> tuple[float, float, float]:
    """Parse only the applied Cartesian target from a receipted proposal."""
    applied = target.applied_xyz
    if len(applied) != 3:
        raise RolloutViolation(RolloutCode.R_NONFINITE, "applied target must have three values")
    x, y, z = (
        _finite_number(applied[0], "applied x"),
        _finite_number(applied[1], "applied y"),
        _finite_number(applied[2], "applied z"),
    )
    if target.clipping_performed:
        raise RolloutViolation(
            RolloutCode.R_CLIPPING_REQUIRED, "upstream Cartesian proposal performed clipping"
        )
    return x, y, z


def degrees_to_radians(degrees: BodyDegrees, domains: Mapping[str, JointDomain]) -> BodyRadians:
    """Affine-map body degrees to mapped radians while rejecting out-of-range input."""
    result: list[float] = []
    for joint, degree in zip(JOINT_ORDER, degrees, strict=True):
        domain = domains[joint]
        p_min, p_max = domain.physical
        q_min, q_max = domain.mapped
        if not p_min <= degree <= p_max:
            raise RolloutViolation(
                RolloutCode.R_OUT_OF_RANGE,
                f"{joint} degree {degree!r} is outside the approved domain",
            )
        mapped = q_min + (degree - p_min) / (p_max - p_min) * (q_max - q_min)
        if not q_min <= mapped <= q_max:
            raise RolloutViolation(
                RolloutCode.R_OUT_OF_RANGE, f"{joint} mapped radian is outside the approved domain"
            )
        result.append(mapped)
    return (result[0], result[1], result[2], result[3], result[4])


def radians_to_degrees(radians: BodyRadians, domains: Mapping[str, JointDomain]) -> BodyDegrees:
    """Invert the affine mapping without any range widening or clipping."""
    result: list[float] = []
    for joint, q_value in zip(JOINT_ORDER, radians, strict=True):
        domain = domains[joint]
        p_min, p_max = domain.physical
        q_min, q_max = domain.mapped
        degree = p_min + (q_value - q_min) / (q_max - q_min) * (p_max - p_min)
        result.append(degree)
    return (result[0], result[1], result[2], result[3], result[4])


def check_elbow_domain(degrees: BodyDegrees, domains: Mapping[str, JointDomain]) -> None:
    """Reject the known invalid elbow even when a Cartesian target is reachable."""
    domain = domains["elbow_flex"]
    p_min, p_max = domain.physical
    q_min, q_max = domain.mapped
    elbow = degrees[2]
    if not p_min <= elbow <= p_max:
        raise RolloutViolation(
            RolloutCode.R_INVALID_ELBOW, f"elbow {elbow!r} is outside the approved domain"
        )
    mapped = q_min + (elbow - p_min) / (p_max - p_min) * (q_max - q_min)
    if not q_min <= mapped <= q_max:
        raise RolloutViolation(
            RolloutCode.R_INVALID_ELBOW, f"elbow mapped radian {mapped!r} is invalid"
        )


@dataclass(frozen=True, slots=True)
class MuJoCoWorkspace:
    """MuJoCo model/data handles plus the pinned end-effector site and Jacobian."""

    scene: Scene
    site_id: int
    jacp: Float64Array
    jacr: Float64Array

    def __post_init__(self) -> None:
        """Validate the five-joint degree indexing the planner depends on."""
        for joint in JOINT_ORDER:
            joint_id = self.scene.mujoco.mj_name2id(
                self.scene.model, self.scene.mujoco.mjtObj.mjOBJ_JOINT, joint
            )
            if int(self.scene.model.jnt_qposadr[joint_id]) != JOINT_ORDER.index(joint):
                raise RolloutViolation(
                    RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "pinned model joint order drifted"
                )


def forward_site(
    workspace: MuJoCoWorkspace,
    radians: Sequence[float],
) -> NDArray[np.float64]:
    """Run FK and return the end-effector site position for the five body joints."""
    scene = workspace.scene
    for index, value in enumerate(radians[:FREE_JOINTS]):
        scene.data.qpos[index] = float(value)
    scene.data.qpos[FREE_JOINTS] = 0.0
    scene.mujoco.mj_forward(scene.model, scene.data)
    return np.array(scene.data.site_xpos[workspace.site_id].copy(), dtype=np.float64)


def site_jacobian(workspace: MuJoCoWorkspace) -> NDArray[np.float64]:
    """Return the 3x5 translational Jacobian of the five body joints."""
    scene = workspace.scene
    scene.mujoco.mj_jacSite(
        scene.model, scene.data, workspace.jacp, workspace.jacr, workspace.site_id
    )
    return workspace.jacp[:, :FREE_JOINTS].copy()


def singularity_metric(jacobian: NDArray[np.float64]) -> float:
    """Return the minimum singular value of the 3x5 translational Jacobian."""
    values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.min(values))
