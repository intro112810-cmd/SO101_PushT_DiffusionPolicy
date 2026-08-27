"""Todo 12: owned no-clipping physical IK planner.

Every RED path returns an exact rejection code before any proposal is promoted;
the one GREEN path round-trips a physical-model FK/IK pose inside every
approved joint domain while keeping gripper authority structurally absent.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.physical_ik import (
    PhysicalIKPlanner,
    PhysicalIKProposal,
    build_physical_ik_planner,
    validate_joint_equivalence_digest,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_collision import pinned_model_digest
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import (
    SceneObjectPoseReceipt,
    ScenePoseExpectations,
    parse_scene_object_pose_receipt,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import FixtureApprovedSafetyPolicy
from so101_pusht_benchmark.sim_to_real.replay_types import JOINT_EQUIVALENCE_DIGEST
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt

BENCHMARK = Path(__file__).resolve().parents[1]
POLICY_PATH = BENCHMARK / "tests/fixtures/sim_to_real/collision_approved_policy.yaml"
POSE_PATH = BENCHMARK / "tests/fixtures/sim_to_real/physical_scene_pose_valid.json"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GREEN_SEED_DEGREES = (
    51.72974687652393,
    -37.76612521912066,
    -55.92287526034229,
    -2.402639880884905,
    167.86029893175413,
)
GREEN_TARGET = (0.02132327532502894, 0.0189276284955288, 0.5176279433055292)
CONTACT_TARGET = (0.196740625, 0.0, 0.02519)
CONTACT_SEED_DEGREES = (0.0, -23.52941176, 49.70414201, 57.32484076, 0.0)
SINGULAR_SEED_DEGREES = (50.15923567, -23.52941176, -74.55621302, -7.1656051, 0.0)
BRANCH_SEED_DEGREES = (0.0, -23.52941176, 49.70414201, 77.32484076, 0.0)
CLIP_TARGET = (0.9, 0.9, 0.5)
UNREACHABLE_TARGET = (0.45, 0.15, 0.06)
OOR_SEED_DEGREES = (0.0, 120.0, 49.70414201, 57.32484076, 0.0)
INVALID_ELBOW_SEED_DEGREES = (0.0, -23.52941176, 106.418, 57.32484076, 0.0)


def _policy() -> FixtureApprovedSafetyPolicy:
    return load_fixture_safety_policy(POLICY_PATH, now=NOW)


def _cartesian(
    xyz: tuple[float, float, float] = GREEN_TARGET,
) -> CartesianProposalReceipt:
    return CartesianProposalReceipt(
        raw_xy=(xyz[0], xyz[1]),
        raw_xyz=xyz,
        applied_xyz=xyz,
        tool_rpy=(0.0, math.pi / 2, 0.0),
        transform_hash="a" * 64,
        camera_digest="b" * 64,
        policy_digest=_policy().canonical_digest,
        clipping_performed=False,
        ik_called=False,
    )


def _planner() -> PhysicalIKPlanner:
    return build_physical_ik_planner()


def _scene_pose(planner: PhysicalIKPlanner) -> SceneObjectPoseReceipt:
    raw = cast("dict[str, object]", json.loads(POSE_PATH.read_text(encoding="utf-8")))
    return parse_scene_object_pose_receipt(
        raw,
        _policy(),
        planner.collision_workspace,
        ScenePoseExpectations(
            cast("str", raw["digest"]),
            "collision-sample-001",
            1000.0,
            "c" * 64,
            "d" * 64,
            "f6453dcc3a48b66d7f7c0f01ea106934eddd196a312e045f93d0fcb0a500fdc3",
            pinned_model_digest(),
            1000.1,
        ),
    )


def _expect(
    code: RolloutCode,
    *,
    target: tuple[float, float, float] = GREEN_TARGET,
    seed: tuple[float, ...] = GREEN_SEED_DEGREES,
    digest: str = JOINT_EQUIVALENCE_DIGEST,
    policy: FixtureApprovedSafetyPolicy | None = None,
) -> None:
    planner = _planner()
    with pytest.raises(RolloutViolation) as caught:
        planner.plan(
            target=_cartesian(target),
            seed_degrees=seed,
            joint_equivalence_digest=digest,
            policy=policy if policy is not None else _policy(),
            scene_pose=_scene_pose(planner),
        )
    assert caught.value.code is code


def test_valid_target_round_trips_without_clipping() -> None:
    planner = _planner()
    proposal = planner.plan(
        target=_cartesian(),
        seed_degrees=GREEN_SEED_DEGREES,
        joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
        policy=_policy(),
        scene_pose=_scene_pose(planner),
    )

    assert isinstance(proposal, PhysicalIKProposal)
    assert len(proposal.body_degrees) == 5
    assert proposal.fk_residual_m <= _policy().kinematics.max_fk_residual_m
    assert proposal.fk_residual_m <= _policy().kinematics.max_ik_residual_m
    assert proposal.singularity_metric >= _policy().kinematics.min_singularity_metric
    assert proposal.branch_delta_degrees <= _policy().kinematics.max_branch_delta_degrees
    assert proposal.clipping_performed is False
    assert proposal.gripper_present is False
    assert proposal.joint_equivalence_digest == JOINT_EQUIVALENCE_DIGEST
    assert len(proposal.proposal_hash) == 64
    assert len(proposal.swept_path) >= 2
    assert all(math.isfinite(value) for value in proposal.body_degrees)
    assert all(math.isfinite(point[0]) for point in proposal.swept_path)
    assert all(math.isfinite(point[1]) for point in proposal.swept_path)
    assert all(math.isfinite(point[2]) for point in proposal.swept_path)


def test_valid_proposal_has_no_gripper_field() -> None:
    planner = _planner()
    proposal = planner.plan(
        target=_cartesian(),
        seed_degrees=GREEN_SEED_DEGREES,
        joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
        policy=_policy(),
        scene_pose=_scene_pose(planner),
    )

    document = proposal.to_document()
    joint_order = document["joint_order"]
    assert isinstance(joint_order, list)
    order = cast("list[str]", joint_order)
    assert set(order) == set(JOINT_ORDER)
    assert "gripper" not in document
    assert document["gripper_present"] is False
    body_degrees = document["body_degrees"]
    assert isinstance(body_degrees, list)
    degrees = cast("list[float]", body_degrees)
    assert len(degrees) == 5


def test_invalid_elbow_blocks_reachable_target() -> None:
    planner = _planner()
    with pytest.raises(RolloutViolation) as caught:
        planner.plan(
            target=_cartesian(),
            seed_degrees=INVALID_ELBOW_SEED_DEGREES,
            joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
            policy=_policy(),
            scene_pose=_scene_pose(planner),
        )
    assert caught.value.code is RolloutCode.R_INVALID_ELBOW


def test_singular_and_branch_jump_reject() -> None:
    _expect(RolloutCode.R_SINGULARITY, target=CONTACT_TARGET, seed=SINGULAR_SEED_DEGREES)
    _expect(
        RolloutCode.R_BRANCH_DISCONTINUITY,
        target=CONTACT_TARGET,
        seed=BRANCH_SEED_DEGREES,
    )


def test_unreachable_target_rejects() -> None:
    _expect(
        RolloutCode.R_IK_UNREACHABLE,
        target=UNREACHABLE_TARGET,
        seed=CONTACT_SEED_DEGREES,
    )


def test_out_of_range_seed_rejects() -> None:
    _expect(RolloutCode.R_OUT_OF_RANGE, seed=OOR_SEED_DEGREES)


def test_clipping_required_target_rejects() -> None:
    _expect(RolloutCode.R_CLIPPING_REQUIRED, target=CLIP_TARGET)


def test_wrong_joint_equivalence_digest_rejects() -> None:
    _expect(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, digest="f" * 64)


def test_wrong_length_seed_rejects() -> None:
    _expect(RolloutCode.R_OUT_OF_RANGE, seed=GREEN_SEED_DEGREES[:4])


def test_gripper_key_in_seed_rejects() -> None:
    planner = _planner()
    with pytest.raises(RolloutViolation) as caught:
        planner.plan(
            target=_cartesian(),
            seed_degrees={"gripper": 0.0},
            joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
            policy=_policy(),
            scene_pose=_scene_pose(planner),
        )
    assert caught.value.code is RolloutCode.R_OUT_OF_RANGE


def test_planner_source_has_no_clip_or_historical_dls_import() -> None:
    from so101_pusht_benchmark.sim_to_real import physical_ik
    from so101_pusht_benchmark.sim_to_real import physical_ik_solve

    for module in (physical_ik, physical_ik_solve):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        assert "np.clip" not in source
        assert ".clip(" not in source
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "so101_pusht_benchmark.sim.dls_ik" not in imported
        assert "dls_ik" not in imported
        assert "SOFollower" not in source
        assert "sync_write" not in source
        assert "Goal_Position" not in source
    assert ".clip(" not in source
    assert "np.clip" not in source


def test_validate_joint_equivalence_digest_is_strict() -> None:
    validate_joint_equivalence_digest(JOINT_EQUIVALENCE_DIGEST)
    with pytest.raises(RolloutViolation) as caught:
        validate_joint_equivalence_digest(JOINT_EQUIVALENCE_DIGEST.upper())
    assert caught.value.code is RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN


def test_proposal_is_immutable() -> None:
    planner = _planner()
    proposal = planner.plan(
        target=_cartesian(),
        seed_degrees=GREEN_SEED_DEGREES,
        joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
        policy=_policy(),
        scene_pose=_scene_pose(planner),
    )

    with pytest.raises(FrozenInstanceError):
        proposal.__setattr__("body_degrees", (0.0, 0.0, 0.0, 0.0, 0.0))
