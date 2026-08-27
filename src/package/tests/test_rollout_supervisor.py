"""Todo 13 guarded first-rollout supervisor contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path

import pytest

from so101_pusht_benchmark.sim_to_real.physical_ik import PhysicalIKProposal
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import FixtureApprovedSafetyPolicy
from so101_pusht_benchmark.sim_to_real.replay_types import (
    CAMERA_REGISTRATION_DIGEST,
    JOINT_EQUIVALENCE_DIGEST,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.rollout_record_types import PhysicalSample
from so101_pusht_benchmark.sim_to_real.single_step_fixture import physical_proposal
from so101_pusht_benchmark.sim_to_real.supervisor import RolloutSupervisor, SupervisorEvidence
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt

LINEAGE_DIGEST = "1" * 64
POLICY_PATH = Path(__file__).parent / "fixtures/sim_to_real/collision_approved_policy.yaml"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
PROPOSAL_HASH = "6d05a6b3dc27ce01deba5804ab176e13c2afc52d5b506c4df582ce77ec9ad3c0"


class FakeWriter:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, *_a: object, **_k: object) -> None:
        self.calls += 1


@dataclass
class FakeClock:
    """Deterministic monotonic clock advanced explicitly by each test."""

    t: float = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _policy() -> FixtureApprovedSafetyPolicy:
    return load_fixture_safety_policy(POLICY_PATH, now=NOW)


def _samples() -> tuple[PhysicalSample, PhysicalSample]:
    return (
        PhysicalSample(
            "sample-1",
            999.95,
            "2" * 64,
            999.94,
            999.95,
            "3" * 64,
            (0.0, -23.5, 49.7, 57.3, 0.0),
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
            (0.1, -23.4, 49.8, 57.2, 0.1),
            "4" * 64,
            "5" * 64,
        ),
    )


def _cartesian(policy: FixtureApprovedSafetyPolicy) -> CartesianProposalReceipt:
    return CartesianProposalReceipt(
        raw_xy=(0.02132327532502894, 0.0189276284955288),
        raw_xyz=(0.02132327532502894, 0.0189276284955288, 0.5176279433055292),
        applied_xyz=(0.02132327532502894, 0.0189276284955288, 0.5176279433055292),
        tool_rpy=(0.0, math.pi / 2, 0.0),
        transform_hash="a" * 64,
        camera_digest=CAMERA_REGISTRATION_DIGEST,
        policy_digest=policy.canonical_digest,
        clipping_performed=False,
        ik_called=False,
    )


def _proposal() -> PhysicalIKProposal:
    return physical_proposal()


def _evidence() -> SupervisorEvidence:
    policy = _policy()
    return SupervisorEvidence(
        lineage_digest=LINEAGE_DIGEST,
        samples=_samples(),
        joint_digest=JOINT_EQUIVALENCE_DIGEST,
        camera_digest=CAMERA_REGISTRATION_DIGEST,
        policy=policy,
        cartesian=_cartesian(policy),
        ik_proposal=_proposal(),
        exclusive_owner=True,
        deadman_active=True,
        stop_clear=True,
        command_id="command-1",
        command_budget=1,
    )


build_evidence = _evidence


def test_complete_evidence_mints_one_token() -> None:
    # Given
    clock = FakeClock()
    writer = FakeWriter()
    supervisor = RolloutSupervisor(clock, writer)

    # When
    token = supervisor.mint(_evidence())

    # Then
    assert token.proposal_hash == PROPOSAL_HASH
    assert token.policy_digest == _policy().canonical_digest
    assert token.command_id == "command-1"
    assert token.valid_until == 1005.0
    assert len(token.digest) == 64
    assert writer.calls == 0


def test_mutated_camera_hash_rejects() -> None:
    # Given
    writer = FakeWriter()
    supervisor = RolloutSupervisor(FakeClock(), writer)
    evidence = replace(
        _evidence(),
        cartesian=replace(_evidence().cartesian, camera_digest="c" * 64),
    )

    # When / Then
    with pytest.raises(RolloutViolation) as caught:
        supervisor.mint(evidence)
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH
    assert writer.calls == 0


EvidenceMutation = Callable[[SupervisorEvidence], SupervisorEvidence]


def _mutate_lineage(value: SupervisorEvidence) -> SupervisorEvidence:
    return replace(value, lineage_digest="0" * 64)


def _mutate_joint_digest(value: SupervisorEvidence) -> SupervisorEvidence:
    return replace(value, joint_digest="0" * 64)


def _mutate_cartesian(value: SupervisorEvidence) -> SupervisorEvidence:
    return replace(value, cartesian=replace(value.cartesian, clipping_performed=True))


def _mutate_deadman(value: SupervisorEvidence) -> SupervisorEvidence:
    return replace(value, deadman_active=False)


def _mutate_budget(value: SupervisorEvidence) -> SupervisorEvidence:
    return replace(value, command_budget=0)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_mutate_lineage, RolloutCode.R_HASH_MISMATCH),
        (_mutate_joint_digest, RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN),
        (_mutate_cartesian, RolloutCode.R_CLIPPING_REQUIRED),
        (_mutate_deadman, RolloutCode.R_DEADMAN_INACTIVE),
        (_mutate_budget, RolloutCode.R_BUDGET_EXHAUSTED),
    ],
    ids=("lineage", "joint-digest", "gripper-omission", "deadman", "budget"),
)
def test_mutated_evidence_rejects_without_writer_call(
    mutate: EvidenceMutation,
    code: RolloutCode,
) -> None:
    # Given
    writer = FakeWriter()
    supervisor = RolloutSupervisor(FakeClock(), writer)

    # When / Then
    with pytest.raises(RolloutViolation) as caught:
        supervisor.mint(mutate(_evidence()))
    assert caught.value.code is code
    assert writer.calls == 0


@pytest.mark.parametrize(
    ("proposal_hash", "command_id"),
    [("c" * 64, "command-1"), (PROPOSAL_HASH, "command-2")],
    ids=("proposal", "command"),
)
def test_token_binding_rejects_other_proposal_or_command(
    proposal_hash: str,
    command_id: str,
) -> None:
    # Given
    writer = FakeWriter()
    supervisor = RolloutSupervisor(FakeClock(), writer)
    token = supervisor.mint(_evidence())

    # When / Then
    with pytest.raises(RolloutViolation) as caught:
        supervisor.consume(token, proposal_hash, command_id)
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH
    assert writer.calls == 0


def test_expired_or_reused_token_rejects() -> None:
    # Given
    expired_clock = FakeClock()
    expired_writer = FakeWriter()
    expired_supervisor = RolloutSupervisor(expired_clock, expired_writer)
    expired = expired_supervisor.mint(_evidence())
    expired_clock.advance(6.0)

    # When / Then
    with pytest.raises(RolloutViolation) as expired_error:
        expired_supervisor.consume(expired, PROPOSAL_HASH, "command-1")
    assert expired_error.value.code is RolloutCode.R_STALE
    assert expired_writer.calls == 0

    # Given
    writer = FakeWriter()
    supervisor = RolloutSupervisor(FakeClock(), writer)
    token = supervisor.mint(_evidence())

    # When
    supervisor.consume(token, PROPOSAL_HASH, "command-1")

    # Then
    with pytest.raises(RolloutViolation) as reused_error:
        supervisor.consume(token, PROPOSAL_HASH, "command-1")
    assert reused_error.value.code is RolloutCode.R_DUPLICATE_DISPATCH
    assert writer.calls == 0
