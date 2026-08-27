"""Collision-proof regression tests for the owned physical IK planner."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from so101_pusht_benchmark.sim_to_real.physical_ik import (
    PhysicalIKPlanner,
    PhysicalIKProposal,
    build_physical_ik_planner,
    physical_ik_proposal_hash,
)
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
from so101_pusht_benchmark.sim_to_real.replay_types import JOINT_EQUIVALENCE_DIGEST
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "tests/fixtures/sim_to_real/collision_approved_policy.yaml"
POSE = ROOT / "tests/fixtures/sim_to_real/physical_scene_pose_valid.json"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
VALID_SEED = (
    51.72974687652393,
    -37.76612521912066,
    -55.92287526034229,
    -2.402639880884905,
    167.86029893175413,
)


class _GeomSizeModel(Protocol):
    geom_size: NDArray[np.float64]


def _policy() -> FixtureApprovedSafetyPolicy:
    return load_fixture_safety_policy(POLICY, now=NOW)


def _scene_pose(
    planner: PhysicalIKPlanner, raw: dict[str, object] | None = None
) -> SceneObjectPoseReceipt:
    document = (
        cast("dict[str, object]", json.loads(POSE.read_text(encoding="utf-8")))
        if raw is None
        else raw
    )
    return parse_scene_object_pose_receipt(
        document,
        _policy(),
        planner.collision_workspace,
        ScenePoseExpectations(
            cast("str", document["digest"]),
            "collision-sample-001",
            1000.0,
            "c" * 64,
            "d" * 64,
            "f6453dcc3a48b66d7f7c0f01ea106934eddd196a312e045f93d0fcb0a500fdc3",
            pinned_model_digest(),
            1000.1,
        ),
    )


def _receipt(xyz: tuple[float, float, float]) -> CartesianProposalReceipt:
    policy = _policy()
    return CartesianProposalReceipt(
        raw_xy=xyz[:2],
        raw_xyz=xyz,
        applied_xyz=xyz,
        tool_rpy=(0.0, math.pi / 2, 0.0),
        transform_hash="a" * 64,
        camera_digest="b" * 64,
        policy_digest=policy.canonical_digest,
        clipping_performed=False,
        ik_called=False,
    )


def test_three_point_path_without_collision_proof_rejects() -> None:
    with pytest.raises(RolloutViolation) as caught:
        PhysicalIKProposal(
            body_degrees=VALID_SEED,
            fk_residual_m=0.0,
            singularity_metric=0.02,
            branch_delta_degrees=0.0,
            swept_path=((0.0, 0.0, 0.1), (0.1, 0.0, 0.1), (0.1, 0.0, 0.1)),
            clipping_performed=False,
            gripper_present=False,
            joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
            proposal_hash="f" * 64,
        )
    assert caught.value.code is RolloutCode.R_COLLISION


def test_held_out_valid_clearance_roundtrip_binds_all_samples_and_digests() -> None:
    planner = build_physical_ik_planner()
    domains = build_joint_domains(_policy())
    radians = degrees_to_radians(VALID_SEED, domains)
    position = forward_site(planner.collision_workspace, radians)
    target = (float(position[0]), float(position[1]), float(position[2]))
    proposal = planner.plan(
        target=_receipt(target),
        seed_degrees=VALID_SEED,
        joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
        policy=_policy(),
        scene_pose=_scene_pose(planner),
    )
    document = proposal.to_document()
    unhashed = dict(document)
    declared = unhashed.pop("proposal_hash")

    assert len(proposal.collision_samples) >= 2
    assert all(sample.digest for sample in proposal.collision_samples)
    assert proposal.policy_digest == _policy().canonical_digest
    assert len(proposal.model_digest) == 64
    assert physical_ik_proposal_hash(unhashed) == declared


def test_obstacle_collision_rejects() -> None:
    planner = build_physical_ik_planner()
    start = degrees_to_radians(VALID_SEED, build_joint_domains(_policy()))
    workspace = planner.collision_workspace
    site = forward_site(workspace, start)
    raw = cast("dict[str, object]", json.loads(POSE.read_text(encoding="utf-8")))
    raw["pusher_transform"] = [
        float(site[0]),
        float(site[1]),
        float(site[2]),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    raw["digest"] = scene_pose_content_digest(raw)
    with pytest.raises(RolloutViolation) as caught:
        swept_collision_proof(
            workspace, start, start, _policy().collision, _scene_pose(planner, raw)
        )
    assert caught.value.code is RolloutCode.R_COLLISION
    assert "object clearance" in str(caught.value)


def test_thin_obstacle_between_clear_endpoints_rejects() -> None:
    planner = build_physical_ik_planner()
    start = degrees_to_radians(VALID_SEED, build_joint_domains(_policy()))
    end = (start[0] + 0.5, start[1], start[2], start[3], start[4])
    workspace = planner.collision_workspace
    midpoint = tuple((left + right) / 2 for left, right in zip(start, end, strict=True))
    midpoint_site = forward_site(workspace, midpoint)
    model = cast("_GeomSizeModel", workspace.scene.model)
    model.geom_size[33, 0] = 0.000001
    raw = cast("dict[str, object]", json.loads(POSE.read_text(encoding="utf-8")))
    raw["pusher_transform"] = [
        float(midpoint_site[0]),
        float(midpoint_site[1]),
        float(midpoint_site[2]),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    raw["digest"] = scene_pose_content_digest(raw)
    pose = _scene_pose(planner, raw)

    swept_collision_proof(workspace, start, start, _policy().collision, pose)
    swept_collision_proof(workspace, end, end, _policy().collision, pose)
    with pytest.raises(RolloutViolation) as caught:
        swept_collision_proof(workspace, start, end, _policy().collision, pose)
    assert caught.value.code is RolloutCode.R_COLLISION


def test_nonadjacent_self_collision_rejects_if_exposed() -> None:
    planner = build_physical_ik_planner()
    domains = build_joint_domains(_policy())
    # This folded pose exposes only non-adjacent self collision in the pinned model.
    folded = degrees_to_radians(
        (
            -41.987957179186715,
            -58.021973972703655,
            -45.95127283224055,
            55.71192613076454,
            -108.1860404076557,
        ),
        domains,
    )
    with pytest.raises(RolloutViolation) as caught:
        swept_collision_proof(
            planner.collision_workspace,
            folded,
            folded,
            _policy().collision,
            _scene_pose(planner),
        )
    assert caught.value.code is RolloutCode.R_COLLISION
    assert "self clearance" in str(caught.value)


def test_threshold_drift_and_sample_hash_mutation_change_proposal_hash(
    tmp_path: Path,
) -> None:
    drifted = tmp_path / "drifted-policy.yaml"
    drifted.write_text(
        POLICY.read_text(encoding="utf-8").replace(
            "minimum_clearance_m: 0.001", "minimum_clearance_m: 0.002"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(drifted, now=NOW)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED

    planner = build_physical_ik_planner()
    radians = degrees_to_radians(VALID_SEED, build_joint_domains(_policy()))
    proof = swept_collision_proof(
        planner.collision_workspace,
        radians,
        radians,
        _policy().collision,
        _scene_pose(planner),
    )
    sample = proof[0]
    changed = replace(sample, minimum_clearance_m=sample.minimum_clearance_m + 1e-9)
    assert changed.digest == sample.digest
    assert changed.valid_digest() is False
    first: dict[str, object] = {
        "collision_samples": [sample.to_document()],
        "policy_digest": _policy().canonical_digest,
    }
    second: dict[str, object] = {
        "collision_samples": [changed.to_document()],
        "policy_digest": _policy().canonical_digest,
    }
    assert physical_ik_proposal_hash(first) != physical_ik_proposal_hash(second)


def test_missing_collision_policy_rejects_as_unauthorized() -> None:
    old_policy = load_fixture_safety_policy(
        ROOT / "tests/fixtures/sim_to_real/approved_policy.yaml", now=NOW
    )
    planner = build_physical_ik_planner()
    radians = degrees_to_radians(VALID_SEED, build_joint_domains(old_policy))
    position = forward_site(planner.collision_workspace, radians)
    target = (float(position[0]), float(position[1]), float(position[2]))
    with pytest.raises(RolloutViolation) as caught:
        planner.plan(
            target=_receipt(target),
            seed_degrees=VALID_SEED,
            joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
            policy=old_policy,
            scene_pose=_scene_pose(planner),
        )
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
