"""Authenticated scene-pose and physical-geometry collision regressions."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from so101_pusht_benchmark.sim_to_real.physical_ik import build_physical_ik_planner
from so101_pusht_benchmark.sim_to_real.physical_ik_collision import (
    pinned_model_digest,
    swept_collision_proof,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import (
    build_joint_domains,
    degrees_to_radians,
    forward_site,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import (
    SceneObjectPoseReceipt,
    ScenePoseExpectations,
    parse_scene_object_pose_receipt,
    scene_pose_content_digest,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import FixtureApprovedSafetyPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).resolve().parents[1]
POSE = ROOT / "tests/fixtures/sim_to_real/physical_scene_pose_valid.json"
POLICY = ROOT / "tests/fixtures/sim_to_real/collision_approved_policy.yaml"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
POSE_DIGEST = "0d795558965afd98ffdfa76a5b5856d12fc9eb68b6a379bea2897ee4aeb0434e"


class _BodyPositionModel(Protocol):
    body_pos: NDArray[np.float64]


CONTACT_SEED = (0.0, -23.52941176, 49.70414201, 57.32484076, 0.0)
SEED = (
    51.72974687652393,
    -37.76612521912066,
    -55.92287526034229,
    -2.402639880884905,
    167.86029893175413,
)


def _raw() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(POSE.read_text(encoding="utf-8")))


def _policy() -> FixtureApprovedSafetyPolicy:
    return load_fixture_safety_policy(POLICY, now=NOW)


def _pose(
    raw: dict[str, object] | None = None,
    *,
    planning_timestamp: float = 1000.1,
    expected_pose_digest: str = POSE_DIGEST,
) -> SceneObjectPoseReceipt:
    planner = build_physical_ik_planner()
    try:
        return parse_scene_object_pose_receipt(
            _raw() if raw is None else raw,
            _policy(),
            planner.collision_workspace,
            ScenePoseExpectations(
                expected_pose_digest,
                "collision-sample-001",
                1000.0,
                "c" * 64,
                "d" * 64,
                "f6453dcc3a48b66d7f7c0f01ea106934eddd196a312e045f93d0fcb0a500fdc3",
                pinned_model_digest(),
                planning_timestamp,
            ),
        )
    finally:
        planner.collision_workspace.scene.close()


def _radians() -> tuple[float, float, float, float, float]:
    return degrees_to_radians(SEED, build_joint_domains(_policy()))


def test_missing_stale_hash_drift_nonfinite_and_domain_pose_reject() -> None:
    planner = build_physical_ik_planner()
    with pytest.raises(RolloutViolation) as missing:
        swept_collision_proof(
            planner.collision_workspace, _radians(), _radians(), _policy().collision, None
        )
    assert missing.value.code is RolloutCode.R_MISSING

    with pytest.raises(RolloutViolation) as stale:
        _pose(planning_timestamp=1000.3)
    assert stale.value.code is RolloutCode.R_STALE

    mutations: tuple[tuple[str, object, RolloutCode], ...] = (
        ("digest", "f" * 64, RolloutCode.R_HASH_MISMATCH),
        ("sample_digest", "e" * 64, RolloutCode.R_HASH_MISMATCH),
        ("pusher_transform", [-0.25, -0.15, math.nan, 1.0, 0.0, 0.0, 0.0], RolloutCode.R_NONFINITE),
        ("push_t_transform", [0.5, 0.16, 0.031, 1.0, 0.0, 0.0, 0.0], RolloutCode.R_OUT_OF_RANGE),
    )
    for key, value, code in mutations:
        raw = deepcopy(_raw())
        raw[key] = value
        with pytest.raises(RolloutViolation) as caught:
            _pose(raw)
        assert caught.value.code is code

    for key, value in (("sample_id", "other-sample"), ("sample_timestamp", 1000.01)):
        changed = deepcopy(_raw())
        changed[key] = value
        changed["digest"] = scene_pose_content_digest(changed)
        with pytest.raises(RolloutViolation) as identity_drift:
            _pose(changed)
        assert identity_drift.value.code is RolloutCode.R_HASH_MISMATCH

    transformed = deepcopy(_raw())
    transformed["pusher_transform"] = [-0.24, -0.15, 0.025, 1.0, 0.0, 0.0, 0.0]
    transformed["digest"] = scene_pose_content_digest(transformed)
    with pytest.raises(RolloutViolation) as transform_drift:
        _pose(transformed)
    assert transform_drift.value.code is RolloutCode.R_HASH_MISMATCH


def test_table_collision_remains_physical() -> None:
    planner = build_physical_ik_planner()
    radians = degrees_to_radians(CONTACT_SEED, build_joint_domains(_policy()))
    with pytest.raises(RolloutViolation) as caught:
        swept_collision_proof(
            planner.collision_workspace, radians, radians, _policy().collision, _pose()
        )
    assert caught.value.code is RolloutCode.R_COLLISION
    assert "table clearance" in str(caught.value)


def test_target_visualization_never_collides_but_physical_pusher_does() -> None:
    planner = build_physical_ik_planner()
    workspace = planner.collision_workspace
    radians = _radians()
    site = forward_site(workspace, radians)
    pose = _pose()

    target_body = workspace.scene.mujoco.mj_name2id(
        workspace.scene.model, workspace.scene.mujoco.mjtObj.mjOBJ_BODY, "target_t"
    )
    model = cast("_BodyPositionModel", workspace.scene.model)
    model.body_pos[target_body] = site
    swept_collision_proof(workspace, radians, radians, _policy().collision, pose)

    raw = _raw()
    raw["pusher_transform"] = [float(site[0]), float(site[1]), float(site[2]), 1.0, 0.0, 0.0, 0.0]
    raw.pop("digest")
    moved_digest = scene_pose_content_digest(raw)
    raw["digest"] = moved_digest
    moved = _pose(raw, expected_pose_digest=moved_digest)
    with pytest.raises(RolloutViolation) as caught:
        swept_collision_proof(workspace, radians, radians, _policy().collision, moved)
    assert caught.value.code is RolloutCode.R_COLLISION
    assert "object clearance" in str(caught.value)
