"""Governed signer and provider identity binding for physical camera corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from .live_capture_identity import ApprovedLiveIdentity, load_approved_live_identity
from .policy_approval import ProductionTrustStore
from .policy_parser import require_production_policy
from .policy_types import ProductionApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation

_SCHEMA = "camera-corpus-authority-v1"
_FIELDS = {
    "schema",
    "artifact_scope",
    "approved_by",
    "approval_id",
    "scheme",
    "corpus_digest",
    "live_identity_digest",
    "provider_digest",
    "profile_digest",
    "camera_device_digest",
    "calibration_digest",
    "orientation_digest",
    "binding_signature",
}


@dataclass(frozen=True, slots=True)
class ProductionCameraAuthority:
    approved_by: str
    approval_id: str
    corpus_digest: str
    identity_digest: str
    provider_digest: str
    camera_device_digest: str
    calibration_digest: str


def _violation(code: RolloutCode, detail: str) -> RolloutViolation:
    return RolloutViolation(code, detail)


def _text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"camera authority {key} invalid")
    return value


def _digest(raw: Mapping[str, object], key: str) -> str:
    value = _text(raw, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise _violation(RolloutCode.R_HASH_MISMATCH, f"camera authority {key} invalid")
    return value


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "signed camera corpus authority is unavailable"
        ) from exc
    if not isinstance(value, dict):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "camera authority must be a mapping")
    result = cast("dict[str, object]", value)
    if set(result) != _FIELDS:
        raise _violation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "camera authority fields are incomplete or unknown",
        )
    return result


def _binding(raw: Mapping[str, object]) -> bytes:
    content = {key: value for key, value in raw.items() if key != "binding_signature"}
    return json.dumps(content, sort_keys=True, separators=(",", ":")).encode()


def verify_production_camera_authority(
    corpus: Mapping[str, object],
    *,
    authority_path: Path,
    identity_path: Path,
    policy: ProductionApprovedSafetyPolicy,
    trust_store: ProductionTrustStore,
) -> tuple[ProductionCameraAuthority, ApprovedLiveIdentity]:
    """Derive physical authority only from one governed signer and exact identities."""
    production_policy = require_production_policy(policy)
    if type(trust_store) is not ProductionTrustStore or not trust_store.is_governed():
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")
    identity = load_approved_live_identity(identity_path, trust_store=trust_store)
    raw = _load(authority_path)
    signer = _text(raw, "approved_by")
    if (
        raw.get("schema") != _SCHEMA
        or raw.get("artifact_scope") != "production"
        or signer != production_policy.approved_by
        or signer != identity.approved_by
    ):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "camera authority scope or owner drift")
    if not trust_store.verify(
        signer,
        _text(raw, "scheme"),
        _binding(raw),
        _text(raw, "binding_signature"),
    ):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "camera corpus authority is untrusted")
    bindings = {
        "live_identity_digest": identity.identity_digest,
        "provider_digest": identity.provider_digest,
        "profile_digest": identity.profile_digest,
        "camera_device_digest": identity.camera_device_digest,
        "calibration_digest": identity.calibration_digest,
        "orientation_digest": corpus.get("orientation_hash"),
        "corpus_digest": corpus.get("camera_digest"),
    }
    if any(_digest(raw, key) != expected for key, expected in bindings.items()):
        raise _violation(RolloutCode.R_HASH_MISMATCH, "camera authority identity drift")
    if (
        corpus.get("device_hash") != identity.camera_device_digest
        or corpus.get("config_hash") != identity.profile_digest
        or corpus.get("resolution") != [identity.camera_width, identity.camera_height]
    ):
        raise _violation(RolloutCode.R_HASH_MISMATCH, "raw corpus provider identity drift")
    return (
        ProductionCameraAuthority(
            signer,
            _text(raw, "approval_id"),
            _digest(raw, "corpus_digest"),
            identity.identity_digest,
            identity.provider_digest,
            identity.camera_device_digest,
            identity.calibration_digest,
        ),
        identity,
    )
