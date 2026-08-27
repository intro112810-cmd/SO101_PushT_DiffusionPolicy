"""Frozen semantic record types for one guarded rollout cycle."""

from __future__ import annotations

from dataclasses import dataclass

BodyDegrees = tuple[float, float, float, float, float]
TargetXY = tuple[float, float]
__all__ = (
    "Acknowledgement",
    "Authorization",
    "BodyDegrees",
    "Command",
    "Evidence",
    "PhysicalSample",
    "PostState",
    "Proposal",
    "RolloutRecord",
    "RolloutRecordVariant",
    "TargetXY",
)


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """Content-addressed immutable record envelope."""

    record_id: str
    created_at: float
    digest: str

    @property
    def identity(self) -> str:
        """Expose the sole identity used to bind downstream records."""
        return self.digest


@dataclass(frozen=True, slots=True)
class PhysicalSample(RolloutRecord):
    camera_timestamp: float
    joint_timestamp: float
    frame_digest: str
    body_degrees: BodyDegrees
    device_digest: str
    calibration_digest: str


@dataclass(frozen=True, slots=True)
class Proposal(RolloutRecord):
    sample_digest: str
    target_xy: TargetXY
    policy_digest: str


@dataclass(frozen=True, slots=True)
class Evidence(RolloutRecord):
    proposal_digest: str
    evidence_type: str
    artifact_digest: str
    valid_until: float


@dataclass(frozen=True, slots=True)
class Authorization(RolloutRecord):
    proposal_digest: str
    evidence_digest: str
    policy_digest: str
    valid_until: float


@dataclass(frozen=True, slots=True)
class Command(RolloutRecord):
    proposal_digest: str
    authorization_digest: str
    body_degrees: BodyDegrees


@dataclass(frozen=True, slots=True)
class Acknowledgement(RolloutRecord):
    command_digest: str
    provider_digest: str
    accepted_body_degrees: BodyDegrees


@dataclass(frozen=True, slots=True)
class PostState(RolloutRecord):
    command_digest: str
    acknowledgement_digest: str
    sample_digest: str
    body_degrees: BodyDegrees


RolloutRecordVariant = (
    PhysicalSample | Proposal | Evidence | Authorization | Command | Acknowledgement | PostState
)
