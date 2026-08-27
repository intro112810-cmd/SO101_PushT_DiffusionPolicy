"""Strict parsing and content identity for immutable rollout records."""

from __future__ import annotations

from collections.abc import Mapping
import math
import string

from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_identity import BoundaryValue, digest_content
from .rollout_record_types import (
    Acknowledgement,
    Authorization,
    BodyDegrees,
    Command,
    Evidence,
    PhysicalSample,
    PostState,
    Proposal,
    RolloutRecordVariant,
    TargetXY,
)

__all__ = (
    "Acknowledgement",
    "Authorization",
    "BoundaryValue",
    "Command",
    "Evidence",
    "PhysicalSample",
    "PostState",
    "Proposal",
    "digest_content",
    "parse_record",
)

_HEX = frozenset(string.hexdigits.lower())
_BASE = frozenset({"kind", "record_id", "created_at", "digest"})
_FIELDS = {
    "physical_sample": frozenset(
        {
            "camera_timestamp",
            "joint_timestamp",
            "frame_digest",
            "body_degrees",
            "device_digest",
            "calibration_digest",
        }
    ),
    "proposal": frozenset({"sample_digest", "target_xy", "policy_digest"}),
    "evidence": frozenset({"proposal_digest", "evidence_type", "artifact_digest", "valid_until"}),
    "authorization": frozenset(
        {"proposal_digest", "evidence_digest", "policy_digest", "valid_until"}
    ),
    "command": frozenset({"proposal_digest", "authorization_digest", "body_degrees"}),
    "acknowledgement": frozenset({"command_digest", "provider_digest", "accepted_body_degrees"}),
    "post_state": frozenset(
        {"command_digest", "acknowledgement_digest", "sample_digest", "body_degrees"}
    ),
}


def _text(value: BoundaryValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_MISSING, label)
    return value


def _number(value: BoundaryValue, label: str) -> float:
    if value is None:
        raise RolloutViolation(RolloutCode.R_MISSING, label)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    result = float(value)
    if not math.isfinite(result):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    return result


def _sha(value: BoundaryValue, label: str) -> str:
    result = _text(value, label).lower()
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return result


def _body(value: BoundaryValue, label: str) -> BodyDegrees:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_MISSING, label)
    sequence = value
    if len(sequence) != 5:
        raise RolloutViolation(RolloutCode.R_MISSING, label)
    return (
        _number(sequence[0], label),
        _number(sequence[1], label),
        _number(sequence[2], label),
        _number(sequence[3], label),
        _number(sequence[4], label),
    )


def _target(value: BoundaryValue) -> TargetXY:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_MISSING, "target_xy")
    sequence = value
    if len(sequence) != 2:
        raise RolloutViolation(RolloutCode.R_MISSING, "target_xy")
    return (_number(sequence[0], "target_xy"), _number(sequence[1], "target_xy"))


def _base(raw: Mapping[str, BoundaryValue], kind: str) -> tuple[str, float, str]:
    expected = _BASE | _FIELDS[kind]
    if frozenset(raw) != expected:
        raise RolloutViolation(RolloutCode.R_MISSING, f"{kind} fields")
    record_id = _text(raw["record_id"], "record_id")
    created_at = _number(raw["created_at"], "created_at")
    digest = _sha(raw["digest"], "digest")
    return record_id, created_at, digest


def _verify(raw: Mapping[str, BoundaryValue], digest: str) -> None:
    content = {key: value for key, value in raw.items() if key != "digest"}
    if digest_content(content) != digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "record content")


def _fresh(timestamp: float, now: float | None, max_age: float | None) -> None:
    if now is not None and max_age is not None and now - timestamp > max_age:
        raise RolloutViolation(RolloutCode.R_STALE, "record timestamp")


def parse_record(
    raw: Mapping[str, BoundaryValue],
    *,
    now: float | None = None,
    max_age: float | None = None,
) -> RolloutRecordVariant:
    """Parse one exhaustive record variant and reject drift before use."""
    kind = _text(raw.get("kind"), "kind")
    if kind not in _FIELDS:
        raise RolloutViolation(RolloutCode.R_MISSING, f"unknown kind {kind}")
    record_id, created_at, digest = _base(raw, kind)
    if now is not None:
        _number(now, "now")
    if max_age is not None and _number(max_age, "max_age") < 0:
        raise RolloutViolation(RolloutCode.R_NONFINITE, "max_age")
    if kind == "physical_sample":
        camera_timestamp = _number(raw["camera_timestamp"], "camera_timestamp")
        joint_timestamp = _number(raw["joint_timestamp"], "joint_timestamp")
        result: RolloutRecordVariant = PhysicalSample(
            record_id,
            created_at,
            digest,
            camera_timestamp,
            joint_timestamp,
            _sha(raw["frame_digest"], "frame_digest"),
            _body(raw["body_degrees"], "body_degrees"),
            _sha(raw["device_digest"], "device_digest"),
            _sha(raw["calibration_digest"], "calibration_digest"),
        )
        freshness_timestamp = min(created_at, camera_timestamp, joint_timestamp)
    elif kind == "proposal":
        result = Proposal(
            record_id,
            created_at,
            digest,
            _sha(raw["sample_digest"], "sample_digest"),
            _target(raw["target_xy"]),
            _sha(raw["policy_digest"], "policy_digest"),
        )
        freshness_timestamp = created_at
    elif kind == "evidence":
        valid_until = _number(raw["valid_until"], "valid_until")
        result = Evidence(
            record_id,
            created_at,
            digest,
            _sha(raw["proposal_digest"], "proposal_digest"),
            _text(raw["evidence_type"], "evidence_type"),
            _sha(raw["artifact_digest"], "artifact_digest"),
            valid_until,
        )
        freshness_timestamp = created_at
        if now is not None and now > valid_until:
            raise RolloutViolation(RolloutCode.R_STALE, "evidence expired")
    elif kind == "authorization":
        valid_until = _number(raw["valid_until"], "valid_until")
        result = Authorization(
            record_id,
            created_at,
            digest,
            _sha(raw["proposal_digest"], "proposal_digest"),
            _sha(raw["evidence_digest"], "evidence_digest"),
            _sha(raw["policy_digest"], "policy_digest"),
            valid_until,
        )
        freshness_timestamp = created_at
        if now is not None and now > valid_until:
            raise RolloutViolation(RolloutCode.R_STALE, "authorization expired")
    elif kind == "command":
        result = Command(
            record_id,
            created_at,
            digest,
            _sha(raw["proposal_digest"], "proposal_digest"),
            _sha(raw["authorization_digest"], "authorization_digest"),
            _body(raw["body_degrees"], "body_degrees"),
        )
        freshness_timestamp = created_at
    elif kind == "acknowledgement":
        result = Acknowledgement(
            record_id,
            created_at,
            digest,
            _sha(raw["command_digest"], "command_digest"),
            _sha(raw["provider_digest"], "provider_digest"),
            _body(raw["accepted_body_degrees"], "accepted_body_degrees"),
        )
        freshness_timestamp = created_at
    else:
        result = PostState(
            record_id,
            created_at,
            digest,
            _sha(raw["command_digest"], "command_digest"),
            _sha(raw["acknowledgement_digest"], "acknowledgement_digest"),
            _sha(raw["sample_digest"], "sample_digest"),
            _body(raw["body_degrees"], "body_degrees"),
        )
        freshness_timestamp = created_at
    _verify(raw, digest)
    _fresh(freshness_timestamp, now, max_age)
    return result
