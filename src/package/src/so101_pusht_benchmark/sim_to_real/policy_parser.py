"""Approved fixture and production policy loading boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .policy_approval import (
    ProductionTrustStore,
    parse_approval_record,
    verify_fixture_approval,
)
from .policy_canonical import CanonicalIdentity, canonical_content, content_digest
from .policy_io import load_yaml_document
from .policy_schema import (
    YamlMapping,
    text_value,
    timestamp_value,
    policy_unauthorized,
)
from .policy_types import (
    FIXTURE_CONSTRUCTION_SEAL,
    PRODUCTION_CONSTRUCTION_SEAL,
    FixtureApprovedSafetyPolicy,
    OwnerApproval,
    ProductionApprovedSafetyPolicy,
    SafetyThresholds,
)
from .policy_values import parse_thresholds_internal

__all__ = (
    "ProductionTrustStore",
    "load_fixture_safety_policy",
    "load_production_safety_policy",
    "require_production_policy",
)
SCHEMA = "so101-sim-to-real-safety-policy-v1"


@dataclass(frozen=True, slots=True)
class _ParsedPolicy:
    policy_id: str
    artifact_scope: str
    approved_by: str
    approved_at: datetime
    valid_from: datetime
    expires_at: datetime
    canonical_content: bytes
    canonical_digest: str
    thresholds: SafetyThresholds
    approval: OwnerApproval
    signed_approval: bytes


def _parse_common(raw: YamlMapping, now: datetime) -> _ParsedPolicy:
    if raw["approval_status"] != "approved":
        raise policy_unauthorized("policy is not owner-approved")
    if raw["schema"] != SCHEMA or raw["policy_version"] != 1:
        raise policy_unauthorized("unsupported policy schema or version")
    policy_id = text_value(raw["policy_id"], "policy_id")
    artifact_scope = text_value(raw["artifact_scope"], "artifact_scope")
    approved_by = text_value(raw["approved_by"], "approved_by")
    approved_at = timestamp_value(raw["approved_at"], "approved_at")
    valid_from = timestamp_value(raw["valid_from"], "valid_from")
    expires_at = timestamp_value(raw["expires_at"], "expires_at")
    if now.tzinfo is None or now.utcoffset() is None:
        raise policy_unauthorized("verification clock must include a timezone")
    thresholds = parse_thresholds_internal(raw)
    invalid_window = (
        approved_at > now or valid_from > now or now >= expires_at or approved_at > valid_from
    )
    if invalid_window:
        raise policy_unauthorized("policy approval or validity window is stale or future")
    if (now - approved_at).total_seconds() > thresholds.timing.max_policy_age_seconds:
        raise policy_unauthorized("policy approval is stale")
    identity = CanonicalIdentity(
        SCHEMA,
        1,
        policy_id,
        artifact_scope,
        approved_by,
        approved_at,
        valid_from,
        expires_at,
    )
    content = canonical_content(identity, thresholds)
    canonical_digest = text_value(raw["canonical_digest"], "canonical_digest")
    if content_digest(content) != canonical_digest:
        raise policy_unauthorized("policy canonical digest drift")
    approval, signed = parse_approval_record(raw)
    if approval.signer_id != approved_by or approval.policy_digest != canonical_digest:
        raise policy_unauthorized("owner approval is not bound to policy digest")
    return _ParsedPolicy(
        policy_id,
        artifact_scope,
        approved_by,
        approved_at,
        valid_from,
        expires_at,
        content,
        canonical_digest,
        thresholds,
        approval,
        signed,
    )


def _clock(now: datetime | None) -> datetime:
    return datetime.now(timezone.utc) if now is None else now


def _fixture_policy(value: _ParsedPolicy) -> FixtureApprovedSafetyPolicy:
    threshold = value.thresholds
    return FixtureApprovedSafetyPolicy(
        FIXTURE_CONSTRUCTION_SEAL,
        SCHEMA,
        1,
        value.policy_id,
        value.artifact_scope,
        value.approved_by,
        value.approved_at,
        value.valid_from,
        value.expires_at,
        value.canonical_content,
        value.canonical_digest,
        threshold.workspace,
        threshold.joint_domains,
        threshold.timing,
        threshold.camera,
        threshold.kinematics,
        threshold.collision,
        threshold.slew,
        threshold.provider,
        threshold.watchdog,
        threshold.acknowledgement,
        threshold.post_state,
        threshold.shadow,
        threshold.single_step,
        threshold.bounded_rollout,
        threshold.operator,
        value.approval,
    )


def _production_policy(value: _ParsedPolicy) -> ProductionApprovedSafetyPolicy:
    threshold = value.thresholds
    return ProductionApprovedSafetyPolicy(
        PRODUCTION_CONSTRUCTION_SEAL,
        SCHEMA,
        1,
        value.policy_id,
        value.artifact_scope,
        value.approved_by,
        value.approved_at,
        value.valid_from,
        value.expires_at,
        value.canonical_content,
        value.canonical_digest,
        threshold.workspace,
        threshold.joint_domains,
        threshold.timing,
        threshold.camera,
        threshold.kinematics,
        threshold.collision,
        threshold.slew,
        threshold.provider,
        threshold.watchdog,
        threshold.acknowledgement,
        threshold.post_state,
        threshold.shadow,
        threshold.single_step,
        threshold.bounded_rollout,
        threshold.operator,
        value.approval,
    )


def load_fixture_safety_policy(
    path: Path,
    *,
    now: datetime | None = None,
) -> FixtureApprovedSafetyPolicy:
    """Load only a pinned, cryptographically approved test fixture."""
    parsed = _parse_common(load_yaml_document(path), _clock(now))
    if parsed.artifact_scope != "test_fixture_only" or not verify_fixture_approval(
        parsed.approval, parsed.signed_approval
    ):
        raise policy_unauthorized("owner approval is not bound to policy digest")
    return _fixture_policy(parsed)


def load_production_safety_policy(
    path: Path,
    *,
    trust_store: ProductionTrustStore,
    now: datetime | None = None,
) -> ProductionApprovedSafetyPolicy:
    """Load production authority only through externally governed anchors."""
    if type(trust_store) is not ProductionTrustStore:
        raise policy_unauthorized("governed production trust store required")
    try:
        governed = trust_store.is_governed()
    except AttributeError as exc:
        raise policy_unauthorized("governed production trust store required") from exc
    if not governed:
        raise policy_unauthorized("governed production trust store required")
    parsed = _parse_common(load_yaml_document(path), _clock(now))
    if parsed.artifact_scope != "production" or not trust_store.verify(
        parsed.approval.signer_id,
        parsed.approval.scheme,
        parsed.signed_approval,
        parsed.approval.binding_signature,
    ):
        raise policy_unauthorized("policy is not approved by a production trust anchor")
    return _production_policy(parsed)


def require_production_policy(value: object) -> ProductionApprovedSafetyPolicy:
    """Reject every fixture, raw, path, boolean, and unsigned value at runtime."""
    if (
        not isinstance(value, ProductionApprovedSafetyPolicy)
        or type(value) is not ProductionApprovedSafetyPolicy
    ):
        raise policy_unauthorized("production-approved safety policy required")
    try:
        approved = value.has_production_authority_marker()
    except AttributeError as exc:
        raise policy_unauthorized("production-approved safety policy required") from exc
    if not approved:
        raise policy_unauthorized("production-approved safety policy required")
    return value
