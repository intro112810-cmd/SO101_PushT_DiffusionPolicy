"""Deterministic fake-bus and supervisor evidence used by rollout fixture QA."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import cast, Final, TypeAlias

from .physical_ik import PhysicalIKProposal, build_physical_ik_planner
from .physical_ik_collision import pinned_model_digest
from .physical_ik_scene_pose import (
    ScenePoseExpectations,
    parse_scene_object_pose_receipt,
)
from .policy_parser import load_fixture_safety_policy
from .policy_types import FixtureApprovedSafetyPolicy
from .replay_types import CAMERA_REGISTRATION_DIGEST, JOINT_EQUIVALENCE_DIGEST
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_record_types import PhysicalSample
from .supervisor import LINEAGE_DIGEST, SupervisorEvidence
from .task_frame_bridge import CartesianProposalReceipt

__all__ = ("FixtureBus", "fixture_evidence", "fixture_policy", "physical_proposal")

_LogPayload: TypeAlias = tuple[str, dict[str, float]] | bool | None
_FIXTURE_NOW: Final = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
_FIXTURES: Final = Path(__file__).resolve().parents[3] / "tests/fixtures/sim_to_real"
_POLICY: Final = _FIXTURES / "collision_approved_policy.yaml"
_SCENE_POSE: Final = _FIXTURES / "single_step_scene_pose.json"


class FixtureBus:
    """Fake direct bus with no acknowledgement or readback authority."""

    def __init__(self) -> None:
        self.log: list[tuple[str, _LogPayload]] = []

    def connect(self) -> None:
        self.log.append(("connect", None))

    def sync_write(self, register: str, payload: dict[str, float]) -> None:
        self.log.append(("sync_write", (register, dict(payload))))

    def disconnect(self, *, disable_torque: bool) -> None:
        self.log.append(("disconnect", disable_torque))

    def ack_payload(self) -> dict[str, float]:
        """Compatibility evidence used only by bounded fixture orchestration."""
        for event, payload in reversed(self.log):
            if event == "sync_write" and isinstance(payload, tuple):
                return dict(payload[1])
        raise RolloutViolation(RolloutCode.R_POST_STATE_MISSING, "no fixture write")


def fixture_policy() -> FixtureApprovedSafetyPolicy:
    if not _POLICY.is_file():
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "fixture policy missing")
    return load_fixture_safety_policy(_POLICY, now=_FIXTURE_NOW)


def _samples() -> tuple[PhysicalSample, PhysicalSample]:
    return (
        PhysicalSample(
            "sample-1",
            999.95,
            "2" * 64,
            999.94,
            999.95,
            "3" * 64,
            (51.7, -37.8, -55.9, -2.4, 167.8),
            "4" * 64,
            "5" * 64,
        ),
        PhysicalSample(
            "sample-2",
            999.97,
            "6" * 64,
            999.96,
            999.97,
            "7" * 64,
            (
                51.72974687652393,
                -37.76612521912066,
                -55.92287526034229,
                -2.402639880884905,
                167.86029893175413,
            ),
            "4" * 64,
            "5" * 64,
        ),
    )


def _cartesian(policy: FixtureApprovedSafetyPolicy) -> CartesianProposalReceipt:
    return CartesianProposalReceipt(
        (0.02132327532502894, 0.0189276284955288),
        (0.02132327532502894, 0.0189276284955288, 0.5176279433055292),
        (0.02132327532502894, 0.0189276284955288, 0.5176279433055292),
        (0.0, math.pi / 2, 0.0),
        "a" * 64,
        CAMERA_REGISTRATION_DIGEST,
        policy.canonical_digest,
        False,
        False,
    )


def physical_proposal() -> PhysicalIKProposal:
    """Plan fixture evidence from an authenticated, sample-bound scene pose."""
    policy = fixture_policy()
    samples = _samples()
    planner = build_physical_ik_planner()
    try:
        raw = json.loads(_SCENE_POSE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RolloutViolation(RolloutCode.R_MISSING, "single-step scene pose mapping")
        document = cast("dict[str, object]", raw)
        pose_digest = document.get("digest")
        if not isinstance(pose_digest, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "single-step scene pose digest")
        second = samples[-1]
        pose = parse_scene_object_pose_receipt(
            document,
            policy,
            planner.collision_workspace,
            ScenePoseExpectations(
                pose_digest,
                second.record_id,
                second.created_at,
                second.digest,
                second.device_digest,
                CAMERA_REGISTRATION_DIGEST,
                pinned_model_digest(),
                second.created_at + 0.01,
            ),
        )
        return planner.plan(
            target=_cartesian(policy),
            seed_degrees=second.body_degrees,
            joint_equivalence_digest=JOINT_EQUIVALENCE_DIGEST,
            policy=policy,
            scene_pose=pose,
        )
    finally:
        planner.collision_workspace.scene.close()


def fixture_evidence() -> SupervisorEvidence:
    """Build deterministic supervisor evidence without granting writer authority."""
    policy = fixture_policy()
    return SupervisorEvidence(
        LINEAGE_DIGEST,
        _samples(),
        JOINT_EQUIVALENCE_DIGEST,
        CAMERA_REGISTRATION_DIGEST,
        policy,
        _cartesian(policy),
        physical_proposal(),
        True,
        True,
        True,
        "command-1",
        1,
    )
