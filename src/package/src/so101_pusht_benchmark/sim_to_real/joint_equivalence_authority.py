"""Governed production authority for physical joint-equivalence corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .joint_equivalence_corpus import digest, mapping, text, unproven
from .policy_approval import ProductionTrustStore
from .policy_types import ProductionApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation

_SCHEMA = "joint-equivalence-corpus-authority-v1"
_FIELDS = frozenset(
    {
        "schema",
        "artifact_scope",
        "approved_by",
        "approval_id",
        "scheme",
        "corpus_digest",
        "policy_digest",
        "provider_digest",
        "device_digest",
        "calibration_digest",
        "capture_id",
        "identity_digest",
        "binding_signature",
    }
)
_IDENTITY_FIELDS = _FIELDS - {"scheme", "identity_digest", "binding_signature"}


@dataclass(frozen=True, slots=True)
class ProductionCorpusAuthority:
    """Verified owner authority bound to exact physical capture identities."""

    corpus_digest: str
    policy_digest: str
    provider_digest: str
    device_digest: str
    calibration_digest: str
    capture_id: str
    identity_digest: str
    approved_by: str
    approval_id: str


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load(path: Path) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise unproven("production corpus authority is absent or malformed") from exc
    result = mapping(raw, "production corpus authority")
    if frozenset(result) != _FIELDS:
        raise unproven("production corpus authority fields are incomplete or unknown")
    return result


def load_production_corpus_authority(
    path: Path,
    *,
    corpus: Mapping[str, object],
    policy: ProductionApprovedSafetyPolicy,
    trust_store: ProductionTrustStore,
) -> ProductionCorpusAuthority:
    """Authenticate detached corpus and provider/device/calibration bindings."""
    if type(trust_store) is not ProductionTrustStore or not trust_store.is_governed():
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")
    raw = _load(path)
    if (
        raw.get("schema") != _SCHEMA
        or raw.get("artifact_scope") != "authorized_physical_diagnostic"
    ):
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "production corpus authority scope is invalid"
        )
    identity_content = {key: raw[key] for key in sorted(_IDENTITY_FIELDS)}
    identity_digest = digest(raw.get("identity_digest"), "corpus identity digest")
    if hashlib.sha256(_canonical(identity_content)).hexdigest() != identity_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "corpus identity digest drift")
    approved_by = text(raw.get("approved_by"), "corpus authority signer")
    approval_id = text(raw.get("approval_id"), "corpus authority approval id")
    binding = _canonical(
        {
            "approval_id": approval_id,
            "identity_digest": identity_digest,
            "schema": _SCHEMA,
            "signer_id": approved_by,
        }
    )
    if not trust_store.verify(
        approved_by,
        text(raw.get("scheme"), "corpus authority scheme"),
        binding,
        text(raw.get("binding_signature"), "corpus authority signature"),
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "production corpus is untrusted")
    authority = ProductionCorpusAuthority(
        digest(raw.get("corpus_digest"), "authority corpus digest"),
        digest(raw.get("policy_digest"), "authority policy digest"),
        digest(raw.get("provider_digest"), "authority provider digest"),
        digest(raw.get("device_digest"), "authority device digest"),
        digest(raw.get("calibration_digest"), "authority calibration digest"),
        text(raw.get("capture_id"), "authority capture id"),
        identity_digest,
        approved_by,
        approval_id,
    )
    if authority.corpus_digest != corpus.get("corpus_digest"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorized corpus digest drift")
    if authority.policy_digest != policy.canonical_digest:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorized policy digest drift")
    bindings = mapping(corpus.get("production_bindings"), "production corpus bindings")
    expected: dict[str, object] = {
        "provider_digest": authority.provider_digest,
        "device_digest": authority.device_digest,
        "calibration_digest": authority.calibration_digest,
        "capture_id": authority.capture_id,
    }
    if dict(bindings) != expected:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "physical capture identity drift")
    return authority
