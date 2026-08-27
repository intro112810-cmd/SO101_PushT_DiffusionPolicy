"""Canonical byte identity built only from fully typed policy values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json

from .policy_types import SafetyThresholds

__all__: tuple[str, ...] = ()


def canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class CanonicalIdentity:
    schema: str
    policy_version: int
    policy_id: str
    artifact_scope: str
    approved_by: str
    approved_at: datetime
    valid_from: datetime
    expires_at: datetime


def canonical_content(identity: CanonicalIdentity, thresholds: SafetyThresholds) -> bytes:
    threshold_content = asdict(thresholds)
    if thresholds.collision is None:
        del threshold_content["collision"]
    content: dict[str, JsonValue] = {
        "schema": identity.schema,
        "policy_version": identity.policy_version,
        "policy_id": identity.policy_id,
        "artifact_scope": identity.artifact_scope,
        "approval_status": "approved",
        "approved_by": identity.approved_by,
        "approved_at": canonical_timestamp(identity.approved_at),
        "valid_from": canonical_timestamp(identity.valid_from),
        "expires_at": canonical_timestamp(identity.expires_at),
        "thresholds": threshold_content,
    }
    return json.dumps(content, sort_keys=True, separators=(",", ":")).encode()


def content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
