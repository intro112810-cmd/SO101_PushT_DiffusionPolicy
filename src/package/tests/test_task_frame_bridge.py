"""Todo 11: transform checkpoint mocap XY into receipted Cartesian proposals.

Every rejection happens before any physical IK planner can run. The bridge
never holds or invokes an IK planner; tests pass a call-counting stub on the
input to prove that failed paths never call ``solve``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import FixtureApprovedSafetyPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame import (
    BOUND_TRANSFORM_HASH,
    CAMERA_DIGEST,
    CANONICAL_SE2,
    PhysicalTableXY,
    SimulatorXY,
    parse_se2_material,
    physical_to_simulator,
    registration_evidence_digest,
    simulator_to_physical,
)
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import (
    BridgeInput,
    CartesianProposalReceipt,
    PreviousAppliedPose,
    build_task_frame_bridge,
    parse_mocap_xy,
)

BENCHMARK = Path(__file__).resolve().parents[1]
POLICY_PATH = BENCHMARK / "tests/fixtures/sim_to_real/approved_policy.yaml"
CAMERA_CORPUS_PATH = BENCHMARK / "tests/fixtures/sim_to_real/camera_registration_valid/corpus.json"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


class _CountingIK(Protocol):
    calls: int

    def solve(self, target: tuple[float, float, float]) -> object: ...


class CountingIK:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, target: tuple[float, float, float]) -> object:
        self.calls += 1
        return {"target": target}


def _policy() -> FixtureApprovedSafetyPolicy:
    return load_fixture_safety_policy(POLICY_PATH, now=NOW)


def _corpus() -> dict[str, object]:
    import json

    raw = json.loads(CAMERA_CORPUS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _mocap(x: float, y: float) -> NDArray[np.float32]:
    return np.asarray([x, y], dtype=np.float32)


def _registered_corpus(
    physical_to_sim: list[float],
    *,
    sim_to_physical: list[float] | None = None,
) -> dict[str, object]:
    corpus = _corpus()
    corpus["physical_to_sim_se2"] = physical_to_sim
    if sim_to_physical is None:
        corpus.pop("sim_to_physical_se2", None)
    else:
        corpus["sim_to_physical_se2"] = sim_to_physical
    corpus["camera_digest"] = registration_evidence_digest(corpus)
    return corpus


def _input(
    *,
    xy: object = _mocap(0.0, 0.0),
    corpus: dict[str, object] | None = None,
    previous: PreviousAppliedPose | None = None,
    ik: _CountingIK | None = None,
    policy: FixtureApprovedSafetyPolicy | None = None,
) -> BridgeInput:
    return BridgeInput(
        camera_corpus=corpus if corpus is not None else _corpus(),
        policy=policy if policy is not None else _policy(),
        raw_xy=parse_mocap_xy(xy),
        previous_applied=previous,
        ik=ik,
    )


def test_registered_xy_emits_receipted_cartesian_proposal() -> None:
    policy = _policy()
    receipt = build_task_frame_bridge(_input(xy=_mocap(0.1, 0.0), policy=policy))

    assert isinstance(receipt, CartesianProposalReceipt)
    assert all(np.isfinite(value) for value in receipt.raw_xyz)
    assert all(np.isfinite(value) for value in receipt.applied_xyz)
    assert receipt.raw_xyz[2] == policy.workspace.contact_z_m
    assert receipt.applied_xyz[0] == pytest.approx(0.1)
    assert receipt.applied_xyz[1] == 0.0
    assert receipt.applied_xyz[2] == 0.025
    assert receipt.tool_rpy == policy.workspace.tool_orientation_rpy_rad
    assert receipt.transform_hash == BOUND_TRANSFORM_HASH
    assert len(receipt.transform_hash) == 64
    assert all(c in "0123456789abcdef" for c in receipt.transform_hash)
    assert receipt.camera_digest == CAMERA_DIGEST
    assert receipt.policy_digest == policy.canonical_digest
    assert receipt.clipping_performed is False
    assert receipt.ik_called is False


def test_identity_round_trip_preserves_in_polygon_point() -> None:
    receipt = build_task_frame_bridge(_input(xy=_mocap(0.2, -0.1)))

    assert receipt.raw_xyz[0] == pytest.approx(0.2)
    assert receipt.raw_xyz[1] == pytest.approx(-0.1)
    assert receipt.raw_xyz[2] == 0.025
    assert receipt.applied_xyz == receipt.raw_xyz


def test_rotation_and_translation_are_inverted_for_simulator_action() -> None:
    # physical_to_sim: s = R(90deg) p + (0.3, -0.1).
    corpus = _registered_corpus(
        [0.0, -1.0, 0.3, 1.0, 0.0, -0.1],
        sim_to_physical=[0.0, 1.0, 0.1, -1.0, 0.0, 0.3],
    )

    receipt = build_task_frame_bridge(_input(xy=_mocap(0.1, 0.0), corpus=corpus))

    assert receipt.raw_xy == pytest.approx((0.1, 0.0))
    assert receipt.raw_xyz == pytest.approx((0.1, 0.2, 0.025))
    assert receipt.applied_xyz == receipt.raw_xyz
    assert receipt.clipping_performed is False
    assert receipt.transform_hash != BOUND_TRANSFORM_HASH
    assert receipt.camera_digest == corpus["camera_digest"]


def test_registered_fiducial_grid_round_trips_between_named_frames() -> None:
    corpus = _registered_corpus([0.0, -1.0, 0.08, 1.0, 0.0, -0.04])
    material = parse_se2_material(corpus)
    fiducials = (
        PhysicalTableXY(-0.2, -0.1),
        PhysicalTableXY(0.0, 0.0),
        PhysicalTableXY(0.2, -0.1),
        PhysicalTableXY(-0.2, 0.1),
        PhysicalTableXY(0.2, 0.1),
    )

    for physical_point in fiducials:
        simulator_point = physical_to_simulator(material, physical_point)
        round_trip = simulator_to_physical(
            material,
            SimulatorXY(simulator_point.x, simulator_point.y),
        )
        assert (round_trip.x, round_trip.y) == pytest.approx(
            (physical_point.x, physical_point.y), abs=1e-12
        )


def test_declared_inverse_mismatch_rejects_before_ik() -> None:
    ik = CountingIK()
    corpus = _registered_corpus(
        [0.0, -1.0, 0.3, 1.0, 0.0, -0.1],
        sim_to_physical=[0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    )

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(corpus=corpus, ik=ik))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID
    assert ik.calls == 0


def test_non_rigid_transform_rejects_without_clipping() -> None:
    corpus = _registered_corpus([1.1, 0.0, 0.0, 0.0, 1.0, 0.0])

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(corpus=corpus))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_held_out_point_maps_to_same_raw_xy() -> None:
    receipt = build_task_frame_bridge(_input(xy=_mocap(-0.05, 0.15)))

    assert receipt.raw_xy[0] == pytest.approx(-0.05)
    assert receipt.raw_xy[1] == pytest.approx(0.15)
    assert receipt.raw_xyz[0] == pytest.approx(-0.05)
    assert receipt.raw_xyz[1] == pytest.approx(0.15)
    assert receipt.raw_xyz[2] == 0.025


def test_out_of_workspace_rejects_before_ik() -> None:
    """A point mapped through the identity SE(2) outside the rectangle rejects."""
    ik = CountingIK()
    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(0.9, 0.0), ik=ik))

    assert caught.value.code is RolloutCode.R_WORKSPACE_VIOLATION
    assert ik.calls == 0


def test_transform_hash_drift_rejects() -> None:
    ik = CountingIK()
    mutated = _corpus()
    mutated["physical_to_sim_se2"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.25]

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(corpus=mutated, ik=ik))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID
    assert ik.calls == 0


def test_camera_digest_byte_mutation_rejects() -> None:
    """A declared camera digest that drifts from the bound digest rejects."""
    mutated = _corpus()
    mutated["camera_digest"] = "b" * 64

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(corpus=mutated))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_se2_byte_mutation_rejects() -> None:
    mutated = _corpus()
    mutated["physical_to_sim_se2"] = [
        CANONICAL_SE2[0],
        CANONICAL_SE2[1],
        CANONICAL_SE2[2] + 0.0001,
        CANONICAL_SE2[3],
        CANONICAL_SE2[4],
        CANONICAL_SE2[5],
    ]

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(corpus=mutated))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_nan_xy_rejects_before_ik() -> None:
    ik = CountingIK()
    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(np.nan, 0.0), ik=ik))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID
    assert ik.calls == 0


def test_infinite_xy_rejects_before_ik() -> None:
    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(np.inf, 0.0)))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_out_of_domain_xy_rejects_before_ik() -> None:
    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(1.5, 0.0)))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_undefined_tool_frame_rejects() -> None:
    """A policy tool orientation with a non-finite component is undefined."""
    policy = _policy()
    object.__setattr__(
        policy.workspace,
        "tool_orientation_rpy_rad",
        (0.0, float("nan"), 0.0),
    )
    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(0.0, 0.0), policy=policy))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_jump_over_slew_rejects_as_clipping_required() -> None:
    ik = CountingIK()
    previous = PreviousAppliedPose(0.0, 0.0, 0.025)
    jump = _policy().slew.max_cartesian_delta_m * 2.0

    with pytest.raises(RolloutViolation) as caught:
        build_task_frame_bridge(_input(xy=_mocap(jump, 0.0), previous=previous, ik=ik))

    assert caught.value.code is RolloutCode.R_CLIPPING_REQUIRED
    assert ik.calls == 0


def test_small_step_does_not_reject() -> None:
    previous = PreviousAppliedPose(0.0, 0.0, 0.025)
    small_step = _policy().slew.max_cartesian_delta_m * 0.5

    receipt = build_task_frame_bridge(_input(xy=_mocap(small_step, 0.0), previous=previous))

    assert receipt.applied_xyz[0] == pytest.approx(small_step)
    assert receipt.applied_xyz[1] == 0.0
    assert receipt.applied_xyz[2] == 0.025


def test_parse_mocap_xy_rejects_non_array() -> None:
    with pytest.raises(RolloutViolation) as caught:
        parse_mocap_xy([0.1, 0.0])

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_parse_mocap_xy_rejects_wrong_dtype() -> None:
    with pytest.raises(RolloutViolation) as caught:
        parse_mocap_xy(np.asarray([0.1, 0.0], dtype=np.float64))

    assert caught.value.code is RolloutCode.R_TRANSFORM_INVALID


def test_receipt_is_immutable() -> None:
    receipt = build_task_frame_bridge(_input(xy=_mocap(0.0, 0.0)))

    with pytest.raises(FrozenInstanceError):
        receipt.__setattr__("applied_xyz", (0.0, 0.0, 0.0))
