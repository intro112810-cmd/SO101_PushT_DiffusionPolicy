"""Owner-approved production identities for synchronized live read providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, InitVar
import hashlib
import json
import math
from pathlib import Path
from typing import cast, Protocol

from .policy_approval import ProductionTrustStore
from .rollout_codes import RolloutCode, RolloutViolation

__all__ = ("ApprovedLiveIdentity", "load_approved_live_identity")
_SCHEMA = "live-read-identity-v1"
_FIELDS = frozenset(
    {
        "schema",
        "artifact_scope",
        "provider_digest",
        "profile_digest",
        "camera_device_digest",
        "follower_device_digest",
        "calibration_digest",
        "camera_width",
        "camera_height",
        "camera_fps",
        "approved_by",
        "approval_id",
        "identity_digest",
        "scheme",
        "binding_signature",
    }
)


class _IdentityConstructionSeal(Protocol):
    """Opaque construction authority held by the verified identity loader."""


PRODUCTION_LIVE_IDENTITY_SEAL: _IdentityConstructionSeal = object()


@dataclass(frozen=True, slots=True)
class ApprovedLiveIdentity:
    """Exact provider, device, calibration, and camera-profile authority."""

    _construction_seal: InitVar[_IdentityConstructionSeal]
    schema: str
    artifact_scope: str
    provider_digest: str
    profile_digest: str
    camera_device_digest: str
    follower_device_digest: str
    calibration_digest: str
    camera_width: int
    camera_height: int
    camera_fps: float
    approved_by: str
    approval_id: str
    identity_digest: str
    _authority_marker: _IdentityConstructionSeal = field(init=False, repr=False, compare=False)

    def __post_init__(self, _construction_seal: _IdentityConstructionSeal) -> None:
        """Reject construction outside the verified production identity loader."""
        if _construction_seal is not PRODUCTION_LIVE_IDENTITY_SEAL:
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED, "live identity is not owner-approved"
            )
        object.__setattr__(self, "_authority_marker", PRODUCTION_LIVE_IDENTITY_SEAL)

    def has_production_authority_marker(self) -> bool:
        return self._authority_marker is PRODUCTION_LIVE_IDENTITY_SEAL


class _VerifyStore(Protocol):
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool: ...

    def is_governed(self) -> bool: ...


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "identity must be a mapping")
    result = cast("Mapping[str, object]", value)
    if frozenset(result) != _FIELDS:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "identity fields are incomplete or unknown"
        )
    return result


def _text(raw: Mapping[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, f"identity {key} is invalid")
    return value


def _digest(raw: Mapping[str, object], key: str) -> str:
    value = _text(raw, key).lower()
    if len(value) != 64:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, f"identity {key} is invalid")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, f"identity {key} is invalid"
        ) from exc
    return value


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, f"identity {key} is invalid")
    return value


def _fps(raw: Mapping[str, object]) -> float:
    value = raw["camera_fps"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "identity camera_fps is invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "identity camera_fps is invalid")
    return result


def _identity_content(raw: Mapping[str, object]) -> bytes:
    payload = {
        key: raw[key]
        for key in sorted(_FIELDS - {"binding_signature", "identity_digest", "scheme"})
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def load_approved_live_identity(
    path: Path,
    *,
    trust_store: ProductionTrustStore,
) -> ApprovedLiveIdentity:
    """Load a production identity only through an owner-governed trust store."""
    if (
        type(trust_store) is not ProductionTrustStore
        or not cast("_VerifyStore", trust_store).is_governed()
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")
    try:
        raw = _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "cannot read live identity evidence"
        ) from exc
    content = _identity_content(raw)
    identity_digest = _digest(raw, "identity_digest")
    if hashlib.sha256(content).hexdigest() != identity_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "live identity digest drift")
    approved_by = _text(raw, "approved_by")
    scheme = _text(raw, "scheme")
    binding = json.dumps(
        {
            "approval_id": _text(raw, "approval_id"),
            "identity_digest": identity_digest,
            "schema": _text(raw, "schema"),
            "signer_id": approved_by,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if not trust_store.verify(approved_by, scheme, binding, _text(raw, "binding_signature")):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "live identity is untrusted")
    if _text(raw, "schema") != _SCHEMA or _text(raw, "artifact_scope") != "production":
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "live identity scope is invalid")
    return ApprovedLiveIdentity(
        PRODUCTION_LIVE_IDENTITY_SEAL,
        _SCHEMA,
        "production",
        _digest(raw, "provider_digest"),
        _digest(raw, "profile_digest"),
        _digest(raw, "camera_device_digest"),
        _digest(raw, "follower_device_digest"),
        _digest(raw, "calibration_digest"),
        _integer(raw, "camera_width"),
        _integer(raw, "camera_height"),
        _fps(raw),
        approved_by,
        _text(raw, "approval_id"),
        identity_digest,
    )
