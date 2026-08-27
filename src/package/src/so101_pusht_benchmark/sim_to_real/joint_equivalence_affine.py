"""Affine order, sign, zero, and scale derivation from measured pose vectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .joint_equivalence_corpus import (
    JointEquivalencePolicy,
    JointMember,
    MAPPING_TOLERANCE,
    unproven,
)
from .joint_mapping import JOINT_ORDER
from .physical_ik_fk import build_joint_domains


@dataclass(frozen=True, slots=True)
class AffineJointMapping:
    """Computed simulator-axis assignment and affine coefficients."""

    joint_order: tuple[str, ...]
    scales_rad_per_degree: tuple[float, ...]
    zero_radians: tuple[float, ...]


def _linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, float, float]:
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0.0:
        return 0.0, y_mean, math.inf
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
        / denominator
    )
    intercept = y_mean - slope * x_mean
    rmse = math.sqrt(
        sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values, strict=True))
        / len(x_values)
    )
    return slope, intercept, rmse


def derive_affine_mapping(
    fit: Sequence[JointMember],
    held_out: Sequence[JointMember],
    policy: JointEquivalencePolicy,
    claimed_order: tuple[str, ...],
) -> AffineJointMapping:
    """Derive axis assignment and coefficients, then validate unseen poses."""
    assignments: list[int] = []
    scales: list[float] = []
    zeros: list[float] = []
    for simulator_axis in range(len(JOINT_ORDER)):
        candidates: list[tuple[float, float, float, int]] = []
        for physical_axis in range(len(JOINT_ORDER)):
            x_values = [member.degrees[physical_axis] for member in fit]
            y_values = [member.radians[simulator_axis] for member in fit]
            candidates.append((*_linear_fit(x_values, y_values), physical_axis))
        slope, intercept, _rmse, physical_axis = min(candidates, key=lambda value: value[2])
        assignments.append(physical_axis)
        scales.append(slope)
        zeros.append(intercept)
    computed_order = tuple(JOINT_ORDER[index] for index in assignments)
    if len(set(assignments)) != len(JOINT_ORDER) or computed_order != claimed_order:
        raise unproven("computed simulator joint order does not match corpus evidence")
    if claimed_order != JOINT_ORDER:
        raise unproven("computed simulator joint order does not match the approved policy")
    domains = build_joint_domains(policy)
    for simulator_axis, physical_axis in enumerate(assignments):
        domain = domains[JOINT_ORDER[physical_axis]]
        expected_scale = (domain.mapped[1] - domain.mapped[0]) / (
            domain.physical[1] - domain.physical[0]
        )
        expected_zero = domain.mapped[0] - expected_scale * domain.physical[0]
        if abs(scales[simulator_axis] - expected_scale) > MAPPING_TOLERANCE:
            raise unproven("computed joint sign/scale does not match approved domains")
        if abs(zeros[simulator_axis] - expected_zero) > MAPPING_TOLERANCE:
            raise unproven("computed joint zero does not match approved domains")
    for member in held_out:
        for simulator_axis, physical_axis in enumerate(assignments):
            predicted = (
                scales[simulator_axis] * member.degrees[physical_axis] + zeros[simulator_axis]
            )
            if abs(predicted - member.radians[simulator_axis]) > MAPPING_TOLERANCE:
                raise unproven("held-out vectors do not validate the computed mapping")
    return AffineJointMapping(computed_order, tuple(scales), tuple(zeros))
