"""Non-actuating gate for one signed, proposal-bound single-step authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

from so101_pusht_benchmark.hardware_profile import load_hardware_profile

from .ledger_chain import canonical_hash, verify_ledger
from .arming_evidence import load_operational_evidence
from .ledger_io import load_ledger_documents
from .policy_approval import ProductionTrustStore
from .policy_parser import load_fixture_safety_policy, load_production_safety_policy
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import SingleStepAuthorization, load_single_step_authorization

__all__ = ("ArmingCheckInput", "ArmingResult", "check_arming", "check_production_arming")


@dataclass(frozen=True, slots=True)
class ArmingCheckInput:
    """Files and verification clock that pin one guarded arming decision."""

    profile_path: Path
    policy_path: Path
    shadow_ledger_path: Path
    authorization_path: Path | None
    operational_evidence_path: Path | None
    now: datetime


@dataclass(frozen=True, slots=True)
class ArmingResult:
    """Content-addressed non-actuating receipt consumed by execution."""

    armed: bool
    motor_writes_performed: bool
    proposal_hash: str
    policy_digest: str
    command_id: str
    shadow_ledger_digest: str
    ownership_digest: str
    interlock_digest: str
    torque_digest: str
    receipt_digest: str
    authorization_digest: str

    def content(self) -> dict[str, object]:
        """Return the exact content signed authorization binds as armed evidence."""
        return {
            "armed": self.armed,
            "command_id": self.command_id,
            "interlock_digest": self.interlock_digest,
            "motor_writes_performed": self.motor_writes_performed,
            "ownership_digest": self.ownership_digest,
            "policy_digest": self.policy_digest,
            "proposal_hash": self.proposal_hash,
            "shadow_ledger_digest": self.shadow_ledger_digest,
            "torque_digest": self.torque_digest,
        }


def _proposal_digests(documents: list[dict[str, object]]) -> frozenset[str]:
    return frozenset(
        digest
        for document in documents
        if document.get("kind") == "ik_proposal"
        and isinstance((digest := document.get("proposal_digest")), str)
        and len(digest) == 64
    )


def _check_arming(
    inputs: ArmingCheckInput,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
    authorization: SingleStepAuthorization,
) -> ArmingResult:
    """Verify exact policy, authorization, shadow, and operational bindings."""
    profile = load_hardware_profile(inputs.profile_path)
    documents = load_ledger_documents(inputs.shadow_ledger_path)
    verify_ledger(documents)
    if profile.policy_digest and profile.policy_digest != policy.canonical_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "profile policy binding")

    if authorization.artifact_scope != policy.artifact_scope:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization scope mismatch")
    if authorization.approved_by != policy.approved_by:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization owner mismatch")
    if authorization.policy_digest != policy.canonical_digest:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization policy mismatch")
    if authorization.proposal_hash not in _proposal_digests(documents):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization proposal mismatch")
    if inputs.operational_evidence_path is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "operational evidence missing")
    operational = load_operational_evidence(
        inputs.operational_evidence_path,
        now=inputs.now,
        max_age_seconds=policy.timing.authorization_max_age_seconds,
    )
    if (
        authorization.ownership_digest != operational.ownership_digest
        or authorization.interlock_digest != operational.interlock_digest
        or authorization.torque_digest != operational.torque_digest
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "operational evidence binding")

    shadow_digest = hashlib.sha256(inputs.shadow_ledger_path.read_bytes()).hexdigest()
    content: dict[str, object] = {
        "armed": True,
        "command_id": authorization.command_id,
        "interlock_digest": operational.interlock_digest,
        "motor_writes_performed": False,
        "ownership_digest": operational.ownership_digest,
        "policy_digest": authorization.policy_digest,
        "proposal_hash": authorization.proposal_hash,
        "shadow_ledger_digest": shadow_digest,
        "torque_digest": operational.torque_digest,
    }
    receipt_digest = canonical_hash(content)
    if authorization.armed_receipt_digest != receipt_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization armed receipt mismatch")
    return ArmingResult(
        armed=True,
        motor_writes_performed=False,
        proposal_hash=authorization.proposal_hash,
        policy_digest=authorization.policy_digest,
        command_id=authorization.command_id,
        shadow_ledger_digest=shadow_digest,
        ownership_digest=operational.ownership_digest,
        interlock_digest=operational.interlock_digest,
        torque_digest=operational.torque_digest,
        receipt_digest=receipt_digest,
        authorization_digest=authorization.digest,
    )


def check_arming(inputs: ArmingCheckInput) -> ArmingResult:
    """Verify fixture arming without granting production authority."""
    if inputs.authorization_path is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "authorization missing")
    policy = load_fixture_safety_policy(inputs.policy_path, now=inputs.now)
    authorization = load_single_step_authorization(inputs.authorization_path, now=inputs.now)
    return _check_arming(inputs, policy, authorization)


def check_production_arming(
    inputs: ArmingCheckInput, trust_store: ProductionTrustStore
) -> ArmingResult:
    """Verify production arming only through one governed owner trust store."""
    if not trust_store.is_governed():
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")
    if inputs.authorization_path is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "authorization missing")
    policy = load_production_safety_policy(
        inputs.policy_path, trust_store=trust_store, now=inputs.now
    )
    authorization = load_single_step_authorization(
        inputs.authorization_path, now=inputs.now, production_verifier=trust_store
    )
    return _check_arming(inputs, policy, authorization)
