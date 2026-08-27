"""Signed owner authorization shared by arming and single-step execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
from typing import Final, Protocol, cast

from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "AuthorizationSignatureVerifier",
    "SingleStepAuthorization",
    "load_single_step_authorization",
)

_SCHEMA: Final = "so101-single-step-authorization-v1"
_FIXTURE_SCOPE: Final = "test_fixture_only"
_FIXTURE_SCHEME: Final = "rsa-pkcs1v15-sha256-test-fixture-v1"
_FIXTURE_SIGNER: Final = "collision-fixture-owner@example.invalid"
_FIXTURE_RSA_EXPONENT: Final = 65537
_FIXTURE_RSA_MODULUS: Final = int(
    "A6615BC739C5B9CB7BB087A1BF8F3EC9E167271AE44C9D03711D9382852E5BABB"
    "53E820508BAA40E82DCA1917478E4EFD43CB4F572B23B5146A31DE12182959126"
    "B635C9541FC56F0515CFB0E6043524B2D8594613983AF4884191C5725D94DC00E"
    "2AB49A69DA3770980E7C97ABFBD5936B68B7AF51EA015C38F201EA6D380CD6E2B"
    "27FC75FB66CFC492E2341B0CC30817A29F7052973D57D21B7D1524249E3C61FA4"
    "DE23F41521AEE9865C266715E1C23D32A9D0CA8FC3C9FD76C59B0739CEE56C2A9"
    "42134499909DB964405D1AA7A177D17F019EACA07480982106203CEB8DB7D8D7C"
    "7D205A891B1F536B7EB84F06CC9B46189141498492534F196D934E1C1",
    16,
)
_FIELDS = frozenset(
    {
        "schema",
        "artifact_scope",
        "approved_by",
        "approved_at",
        "expires_at",
        "policy_digest",
        "proposal_hash",
        "command_id",
        "command_budget",
        "ownership_digest",
        "interlock_digest",
        "torque_digest",
        "armed_receipt_digest",
        "signature_scheme",
        "signer_id",
        "approval_id",
        "digest",
        "binding_signature",
    }
)
_HEX = frozenset("0123456789abcdef")


class AuthorizationSignatureVerifier(Protocol):
    """Externally governed verifier for a production authorization signature."""

    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        """Return whether an owner-controlled anchor verifies the content."""
        ...


@dataclass(frozen=True, slots=True)
class SingleStepAuthorization:
    """Authenticated, expiring authority for exactly one armed proposal."""

    artifact_scope: str
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    policy_digest: str
    proposal_hash: str
    command_id: str
    command_budget: int
    ownership_digest: str
    interlock_digest: str
    torque_digest: str
    armed_receipt_digest: str
    signature_scheme: str
    signer_id: str
    approval_id: str
    digest: str


def _violation(code: RolloutCode, detail: str) -> RolloutViolation:
    return RolloutViolation(code, detail)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization must be a JSON object")
    return cast("dict[str, object]", value)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeDecodeError) as exc:
        raise _violation(RolloutCode.R_MISSING, f"authorization missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authorization is not valid JSON"
        ) from exc
    document = _mapping(value)
    if frozenset(document) != _FIELDS:
        raise _violation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "authorization fields are incomplete or unknown",
        )
    return document


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"duplicate key: {key}")
        result[key] = value
    return result


def _text(document: Mapping[str, object], key: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} invalid")
    return value


def _digest(document: Mapping[str, object], key: str) -> str:
    value = _text(document, key)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise _violation(RolloutCode.R_HASH_MISMATCH, f"authorization {key} invalid")
    return value


def _timestamp(document: Mapping[str, object], key: str) -> datetime:
    try:
        value = datetime.fromisoformat(_text(document, key).replace("Z", "+00:00"))
    except ValueError as exc:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} invalid") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} lacks timezone")
    return value


def _signed_content(document: Mapping[str, object]) -> bytes:
    content = {
        key: value for key, value in document.items() if key not in {"digest", "binding_signature"}
    }
    return json.dumps(content, sort_keys=True, separators=(",", ":")).encode()


def _verify_fixture_signature(content: bytes, signature_hex: str) -> bool:
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    size = (_FIXTURE_RSA_MODULUS.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    encoded = pow(
        int.from_bytes(signature, "big"), _FIXTURE_RSA_EXPONENT, _FIXTURE_RSA_MODULUS
    ).to_bytes(size, "big")
    digest_info = (
        bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(content).digest()
    )
    expected = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    return hmac.compare_digest(encoded, expected)


def load_single_step_authorization(
    path: Path,
    *,
    now: datetime,
    production_verifier: AuthorizationSignatureVerifier | None = None,
) -> SingleStepAuthorization:
    """Parse and authenticate one signed authorization for the supplied clock."""
    document = _load(path)
    if _text(document, "schema") != _SCHEMA:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization schema invalid")
    content = _signed_content(document)
    if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), _digest(document, "digest")):
        raise _violation(RolloutCode.R_HASH_MISMATCH, "authorization content digest mismatch")
    approved_at = _timestamp(document, "approved_at")
    expires_at = _timestamp(document, "expires_at")
    if now.tzinfo is None or now.utcoffset() is None:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization clock lacks timezone")
    if approved_at > now or now >= expires_at or approved_at >= expires_at:
        raise _violation(RolloutCode.R_STALE, "authorization is expired or not yet valid")
    budget = document["command_budget"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget != 1:
        raise _violation(RolloutCode.R_BUDGET_EXHAUSTED, "one-call authorization required")
    authorization = SingleStepAuthorization(
        artifact_scope=_text(document, "artifact_scope"),
        approved_by=_text(document, "approved_by"),
        approved_at=approved_at,
        expires_at=expires_at,
        policy_digest=_digest(document, "policy_digest"),
        proposal_hash=_digest(document, "proposal_hash"),
        command_id=_text(document, "command_id"),
        command_budget=budget,
        ownership_digest=_digest(document, "ownership_digest"),
        interlock_digest=_digest(document, "interlock_digest"),
        torque_digest=_digest(document, "torque_digest"),
        armed_receipt_digest=_digest(document, "armed_receipt_digest"),
        signature_scheme=_text(document, "signature_scheme"),
        signer_id=_text(document, "signer_id"),
        approval_id=_text(document, "approval_id"),
        digest=_digest(document, "digest"),
    )
    signature = _text(document, "binding_signature")
    fixture_valid = (
        authorization.artifact_scope == _FIXTURE_SCOPE
        and authorization.signature_scheme == _FIXTURE_SCHEME
        and authorization.signer_id == _FIXTURE_SIGNER
        and _verify_fixture_signature(content, signature)
    )
    production_valid = (
        authorization.artifact_scope == "production"
        and production_verifier is not None
        and production_verifier.verify(
            authorization.signer_id, authorization.signature_scheme, content, signature
        )
    )
    if not fixture_valid and not production_valid:
        raise _violation(RolloutCode.R_HASH_MISMATCH, "authorization signature invalid")
    if authorization.signer_id != authorization.approved_by:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization owner mismatch")
    return authorization
