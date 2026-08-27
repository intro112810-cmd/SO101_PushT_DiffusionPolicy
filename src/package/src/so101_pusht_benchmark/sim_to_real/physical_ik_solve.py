"""Damped least-squares body-only solve with fail-closed domain checks.

Uses the DLS *math* from the historical oracle only; it never imports that
oracle, never clamps a joint, never widens a joint range, and never emits a
gripper value. A step that would leave the approved mapped-radian domain
raises ``R_CLIPPING_REQUIRED`` instead of clamping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import (
    BodyRadians,
    MuJoCoWorkspace,
    forward_site,
    site_jacobian,
    singularity_metric,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SolveParams:
    """Solver budget and policy thresholds; no value may be absent or non-finite."""

    max_iterations: int
    tolerance_m: float
    damping: float
    min_singularity_metric: float

    def __post_init__(self) -> None:
        """Reject absent or non-finite solver/policy inputs at construction."""
        if self.max_iterations <= 0:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "iteration budget invalid")
        if not math.isfinite(self.tolerance_m) or self.tolerance_m <= 0.0:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "IK tolerance invalid")
        if not math.isfinite(self.damping) or self.damping <= 0.0:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "DLS damping invalid")
        if not math.isfinite(self.min_singularity_metric) or self.min_singularity_metric <= 0.0:
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED, "singularity threshold invalid"
            )


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    """Converged or rejected solve without any clipping authority."""

    converged: bool
    code: RolloutCode | None
    radians: BodyRadians
    residual_m: float
    singular_value: float
    path: tuple[tuple[float, float, float], ...]


def _step_candidate(
    jacobian: Float64Array,
    error: Float64Array,
    damping: float,
) -> Float64Array:
    gram = jacobian @ jacobian.T
    regularized = gram + damping * np.eye(3)
    dual = np.linalg.solve(regularized, error)
    return jacobian.T @ dual


def solve_dls(
    target: tuple[float, float, float],
    seed_radians: BodyRadians,
    domains: dict[str, tuple[float, float]],
    workspace: MuJoCoWorkspace,
    params: SolveParams,
) -> SolveOutcome:
    """Iterate DLS on the five body joints, rejecting instead of clipping."""
    current = np.asarray(seed_radians, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    seed_path: list[tuple[float, float, float]] = [tuple(target_array.tolist())]
    iteration = 0
    residual = math.inf
    singular_value = 0.0
    while iteration < params.max_iterations:
        iteration += 1
        position = forward_site(workspace, current.tolist())
        error = target_array - position
        residual = float(np.linalg.norm(error))
        jacobian = site_jacobian(workspace)
        singular_value = singularity_metric(jacobian)
        if residual <= params.tolerance_m:
            radians: BodyRadians = (
                float(current[0]),
                float(current[1]),
                float(current[2]),
                float(current[3]),
                float(current[4]),
            )
            return SolveOutcome(
                True,
                None,
                radians,
                residual,
                singular_value,
                tuple(seed_path),
            )
        if singular_value < params.min_singularity_metric:
            radians = (
                float(current[0]),
                float(current[1]),
                float(current[2]),
                float(current[3]),
                float(current[4]),
            )
            return SolveOutcome(
                False,
                RolloutCode.R_SINGULARITY,
                radians,
                residual,
                singular_value,
                tuple(seed_path),
            )
        step = _step_candidate(jacobian, error, params.damping)
        candidate = current + step
        for joint, value in zip(JOINT_ORDER, candidate.tolist(), strict=True):
            q_min, q_max = domains[joint]
            if not q_min <= value <= q_max:
                radians = (
                    float(candidate[0]),
                    float(candidate[1]),
                    float(candidate[2]),
                    float(candidate[3]),
                    float(candidate[4]),
                )
                return SolveOutcome(
                    False,
                    RolloutCode.R_CLIPPING_REQUIRED,
                    radians,
                    residual,
                    singular_value,
                    tuple(seed_path),
                )
        current = candidate

    radians = (
        float(current[0]),
        float(current[1]),
        float(current[2]),
        float(current[3]),
        float(current[4]),
    )
    return SolveOutcome(
        False,
        RolloutCode.R_IK_UNREACHABLE,
        radians,
        residual,
        singular_value,
        tuple(seed_path),
    )
