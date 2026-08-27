"""Nominal immutable safety-policy authority contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from typing import Protocol

__all__ = ("FixtureApprovedSafetyPolicy", "ProductionApprovedSafetyPolicy")


class _ConstructionSeal(Protocol):
    """Opaque capability token shared by the fixture and production parsers."""


class PolicyConstructionError(ValueError):
    """Construction attempted outside the pinned policy parser boundary."""


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class WorkspacePolicy:
    polygon_xy_m: tuple[tuple[float, float], ...]
    contact_z_m: float
    tool_orientation_rpy_rad: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class JointDomains:
    joint_order: tuple[str, ...]
    physical_degrees: tuple[NumericRange, ...]
    mapped_radians: tuple[NumericRange, ...]


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    sample_max_age_seconds: float
    sample_max_skew_seconds: float
    max_policy_age_seconds: float
    authorization_max_age_seconds: float
    authorization_ttl_seconds: float


@dataclass(frozen=True, slots=True)
class CameraPolicy:
    max_reprojection_error_px: float
    min_correspondences: int
    max_correspondence_error_px: float


@dataclass(frozen=True, slots=True)
class KinematicsPolicy:
    max_fk_residual_m: float
    max_ik_residual_m: float
    min_singularity_metric: float
    max_branch_delta_degrees: float


@dataclass(frozen=True, slots=True)
class CollisionPolicy:
    minimum_clearance_m: float
    max_joint_step_radians: float
    max_path_samples: int


@dataclass(frozen=True, slots=True)
class SlewPolicy:
    max_cartesian_delta_m: float
    max_joint_delta_degrees: float


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    exact_goal_required: bool
    max_abs_error_degrees: float


@dataclass(frozen=True, slots=True)
class WatchdogPolicy:
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AcknowledgementPolicy:
    required: bool
    timeout_seconds: float
    max_position_error_degrees: float


@dataclass(frozen=True, slots=True)
class PostStatePolicy:
    max_age_seconds: float
    max_tracking_error_degrees: float


@dataclass(frozen=True, slots=True)
class ShadowBudget:
    min_cycles: int
    max_cycle_latency_seconds: float
    max_error_count: int


@dataclass(frozen=True, slots=True)
class SingleStepBudget:
    max_commands: int


@dataclass(frozen=True, slots=True)
class BoundedRolloutBudget:
    max_commands: int
    max_duration_seconds: float
    max_path_length_m: float
    max_error_count: int


@dataclass(frozen=True, slots=True)
class OperatorPolicy:
    deadman_required: bool
    stop_required: bool
    stop_behavior: str
    acknowledgement_required: bool


@dataclass(frozen=True, slots=True)
class SafetyThresholds:
    workspace: WorkspacePolicy
    joint_domains: JointDomains
    timing: TimingPolicy
    camera: CameraPolicy
    kinematics: KinematicsPolicy
    collision: CollisionPolicy | None
    slew: SlewPolicy
    provider: ProviderPolicy
    watchdog: WatchdogPolicy
    acknowledgement: AcknowledgementPolicy
    post_state: PostStatePolicy
    shadow: ShadowBudget
    single_step: SingleStepBudget
    bounded_rollout: BoundedRolloutBudget
    operator: OperatorPolicy


@dataclass(frozen=True, slots=True)
class OwnerApproval:
    scheme: str
    approval_id: str
    signer_id: str
    policy_digest: str
    binding_signature: str


FIXTURE_CONSTRUCTION_SEAL: _ConstructionSeal = object()
PRODUCTION_CONSTRUCTION_SEAL: _ConstructionSeal = object()


@dataclass(frozen=True, slots=True)
class _ApprovedSafetyPolicyFields:
    _construction_seal: InitVar[_ConstructionSeal]
    schema: str
    policy_version: int
    policy_id: str
    artifact_scope: str
    approved_by: str
    approved_at: datetime
    valid_from: datetime
    expires_at: datetime
    canonical_content: bytes
    canonical_digest: str
    workspace: WorkspacePolicy
    joint_domains: JointDomains
    timing: TimingPolicy
    camera: CameraPolicy
    kinematics: KinematicsPolicy
    collision: CollisionPolicy | None
    slew: SlewPolicy
    provider: ProviderPolicy
    watchdog: WatchdogPolicy
    acknowledgement: AcknowledgementPolicy
    post_state: PostStatePolicy
    shadow: ShadowBudget
    single_step: SingleStepBudget
    bounded_rollout: BoundedRolloutBudget
    operator: OperatorPolicy
    owner_approval: OwnerApproval
    _authority_marker: _ConstructionSeal = field(init=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FixtureApprovedSafetyPolicy(_ApprovedSafetyPolicyFields):
    """Cryptographically approved authority restricted to test fixtures."""

    def __post_init__(self, _construction_seal: _ConstructionSeal) -> None:
        """Reject construction outside the pinned fixture parser."""
        if _construction_seal is not FIXTURE_CONSTRUCTION_SEAL:
            raise PolicyConstructionError("fixture policy construction is parser-private")
        object.__setattr__(self, "_authority_marker", FIXTURE_CONSTRUCTION_SEAL)


@dataclass(frozen=True, slots=True)
class ProductionApprovedSafetyPolicy(_ApprovedSafetyPolicyFields):
    """Production authority constructible only after a trusted production approval."""

    def __post_init__(self, _construction_seal: _ConstructionSeal) -> None:
        """Reject construction outside the production trust boundary."""
        if _construction_seal is not PRODUCTION_CONSTRUCTION_SEAL:
            raise PolicyConstructionError("production policy construction is parser-private")
        object.__setattr__(self, "_authority_marker", PRODUCTION_CONSTRUCTION_SEAL)

    def has_production_authority_marker(self) -> bool:
        return self._authority_marker is PRODUCTION_CONSTRUCTION_SEAL
