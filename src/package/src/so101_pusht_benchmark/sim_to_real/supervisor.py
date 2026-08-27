"""Independent evidence gate for one guarded physical rollout proposal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
import math
import re
from threading import Lock
from typing import Final, Protocol

from .authorization import (
    AuthorizationClaim,
    AuthorizationToken,
    mint_authorization,
    verify_authorization,
)
from .physical_ik import PhysicalIKProposal
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .replay_types import CAMERA_REGISTRATION_DIGEST, JOINT_EQUIVALENCE_DIGEST
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_record_types import PhysicalSample
from .sample_capture import Clock
from .task_frame import CartesianPoint3, check_workspace_violation
from .task_frame_bridge import CartesianProposalReceipt

__all__ = ("LINEAGE_DIGEST", "AuthorizationToken", "RolloutSupervisor", "SupervisorEvidence")
LINEAGE_DIGEST: Final = "1" * 64
_COMMAND_PATTERN: Final = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.ASCII)


class Writer(Protocol):
    """Compatibility seam deliberately unused until the Todo 16 write boundary."""


@dataclass(frozen=True, slots=True)
class SupervisorEvidence:
    """All read-only receipts required to authorize one proposal."""

    lineage_digest: str
    samples: tuple[PhysicalSample, PhysicalSample]
    joint_digest: str
    camera_digest: str
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy
    cartesian: CartesianProposalReceipt
    ik_proposal: PhysicalIKProposal
    exclusive_owner: bool
    deadman_active: bool
    stop_clear: bool
    command_id: str
    command_budget: int


class RolloutSupervisor:
    """Thread-safe one-use authorization state; it has no actuation capability."""

    def __init__(self, clock: Clock, writer: Writer | None = None) -> None:
        del writer
        self._clock = clock
        self._issued: set[str] = set()
        self._consumed: set[str] = set()
        self._lock = Lock()

    def mint(self, evidence: SupervisorEvidence) -> AuthorizationToken:
        """Fail closed unless every receipt authorizes exactly one fresh command."""
        now = _now(self._clock())
        _validate_evidence(evidence, now)
        claim = AuthorizationClaim(
            proposal_hash=evidence.ik_proposal.proposal_hash,
            policy_digest=evidence.policy.canonical_digest,
            command_id=evidence.command_id,
            valid_until=now + evidence.policy.timing.authorization_ttl_seconds,
        )
        token = mint_authorization(claim)
        with self._lock:
            self._issued.add(token.digest)
        return token

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        """Consume one minted token without invoking a physical writer."""
        verify_authorization(token)
        if _now(self._clock()) > token.valid_until:
            raise RolloutViolation(RolloutCode.R_STALE, "authorization expired")
        if proposal_hash != token.proposal_hash or command_id != token.command_id:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization binding")
        with self._lock:
            if token.digest not in self._issued:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "unissued authorization")
            if token.digest in self._consumed:
                raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, "authorization consumed")
            self._consumed.add(token.digest)


def _now(value: float) -> float:
    if not math.isfinite(value):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "supervisor clock")
    return value


def _validate_evidence(evidence: SupervisorEvidence, now: float) -> None:
    if type(evidence.policy) is not FixtureApprovedSafetyPolicy:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "actuation supervisor requires its exact guarded-rollout policy type",
        )
    if not evidence.exclusive_owner:
        raise RolloutViolation(RolloutCode.R_OWNERSHIP_CONFLICT, "exclusive writer ownership")
    if not evidence.deadman_active or not evidence.stop_clear:
        raise RolloutViolation(RolloutCode.R_DEADMAN_INACTIVE, "deadman or stop gate")
    if evidence.command_budget != 1:
        raise RolloutViolation(RolloutCode.R_BUDGET_EXHAUSTED, "one-command budget required")
    if evidence.lineage_digest != LINEAGE_DIGEST:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "lineage receipt")
    if evidence.joint_digest != JOINT_EQUIVALENCE_DIGEST:
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint receipt")
    if (
        evidence.camera_digest != CAMERA_REGISTRATION_DIGEST
        or evidence.cartesian.camera_digest != CAMERA_REGISTRATION_DIGEST
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "camera receipt")
    if evidence.policy.canonical_digest != evidence.cartesian.policy_digest:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "Cartesian policy receipt")
    _validate_cartesian(evidence.cartesian, evidence.policy)
    _validate_proposal(evidence.ik_proposal, evidence.policy)
    _validate_samples(evidence.samples, evidence.policy, now)
    _command_id(evidence.command_id)


def _validate_cartesian(
    receipt: CartesianProposalReceipt,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> None:
    if receipt.clipping_performed or receipt.ik_called or receipt.raw_xyz != receipt.applied_xyz:
        raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "Cartesian proposal transformed")
    coordinates = (*receipt.raw_xyz, *receipt.applied_xyz, *receipt.tool_rpy)
    if not all(math.isfinite(value) for value in coordinates):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "Cartesian receipt")
    check_workspace_violation(
        policy.workspace.polygon_xy_m,
        CartesianPoint3(*receipt.applied_xyz),
    )


def _validate_proposal(
    proposal: PhysicalIKProposal,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> None:
    if proposal.clipping_performed:
        raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "IK proposal clipped")
    values = (
        *proposal.body_degrees,
        proposal.fk_residual_m,
        proposal.singularity_metric,
        proposal.branch_delta_degrees,
    )
    if not all(math.isfinite(value) for value in values):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "IK proposal")
    if proposal.gripper_present or len(proposal.body_degrees) != 5:
        raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "body-only proposal required")
    if proposal.joint_equivalence_digest != JOINT_EQUIVALENCE_DIGEST:
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "proposal joint receipt")
    if proposal.fk_residual_m > policy.kinematics.max_ik_residual_m:
        raise RolloutViolation(RolloutCode.R_IK_UNREACHABLE, "IK residual")
    if proposal.singularity_metric < policy.kinematics.min_singularity_metric:
        raise RolloutViolation(RolloutCode.R_SINGULARITY, "IK singularity")
    if proposal.branch_delta_degrees > policy.kinematics.max_branch_delta_degrees:
        raise RolloutViolation(RolloutCode.R_BRANCH_DISCONTINUITY, "IK branch")
    for degree, domain in zip(
        proposal.body_degrees,
        policy.joint_domains.physical_degrees,
        strict=True,
    ):
        if degree < domain.minimum or degree > domain.maximum:
            raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "body joint domain")
    _validate_swept_path(proposal.swept_path, policy)


def _validate_swept_path(
    path: Sequence[tuple[float, float, float]],
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> None:
    if len(path) < 2 or any(not all(math.isfinite(value) for value in point) for point in path):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "IK swept path")
    jumps = (math.dist(previous, current) for previous, current in pairwise(path))
    if any(jump > policy.slew.max_cartesian_delta_m for jump in jumps):
        raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "IK swept path slew")


def _validate_samples(
    samples: tuple[PhysicalSample, PhysicalSample],
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
    now: float,
) -> None:
    if len(samples) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "sample count")
    first, second = samples
    if len({sample.record_id for sample in samples}) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "sample record_id")
    if len({sample.digest for sample in samples}) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "sample digest")
    if len({sample.frame_digest for sample in samples}) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "sample frame")
    for sample in samples:
        timestamps = (sample.created_at, sample.camera_timestamp, sample.joint_timestamp)
        if not all(math.isfinite(timestamp) for timestamp in timestamps):
            raise RolloutViolation(RolloutCode.R_STALE, "sample timestamp")
        if (
            abs(sample.camera_timestamp - sample.joint_timestamp)
            > policy.timing.sample_max_skew_seconds
        ):
            raise RolloutViolation(RolloutCode.R_STALE, "sample skew")
        if any(
            timestamp > now or now - timestamp > policy.timing.sample_max_age_seconds
            for timestamp in timestamps
        ):
            raise RolloutViolation(RolloutCode.R_STALE, "sample age")
    if (
        second.created_at <= first.created_at
        or second.camera_timestamp <= first.camera_timestamp
        or second.joint_timestamp <= first.joint_timestamp
    ):
        raise RolloutViolation(RolloutCode.R_STALE, "sample timestamps")


def _command_id(value: str) -> str:
    if len(value) > 128 or _COMMAND_PATTERN.fullmatch(value) is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "invalid command_id")
    return value
