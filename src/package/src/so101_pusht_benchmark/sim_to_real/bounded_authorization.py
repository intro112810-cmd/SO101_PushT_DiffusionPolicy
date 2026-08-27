"""Signed authority and verified single-step promotion for bounded rollout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
from typing import Final, Protocol, cast

from .bounded_promotion import verify_single_step_receipt_document
from .rollout_codes import RolloutCode, RolloutViolation

_SCHEMA: Final = "so101-bounded-rollout-authorization-v1"
_PRODUCTION_SCHEMA: Final = "so101-bounded-rollout-authorization-v2"
_SCOPE: Final = "test_fixture_only"
_SCHEME: Final = "rsa-pkcs1v15-sha256-test-fixture-v1"
_SIGNER: Final = "collision-fixture-owner@example.invalid"
_EXPONENT: Final = 65537
_MODULUS: Final = int(
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
_V1_FIELDS = frozenset(
    {
        "schema",
        "artifact_scope",
        "approved_by",
        "approved_at",
        "expires_at",
        "policy_digest",
        "single_step_receipt_digest",
        "max_commands",
        "max_duration_seconds",
        "max_path_length_m",
        "max_error_count",
        "signature_scheme",
        "signer_id",
        "approval_id",
        "digest",
        "binding_signature",
    }
)
_V2_FIELDS = _V1_FIELDS | {"cycle_provider_digest"}
_HEX = frozenset("0123456789abcdef")


class BoundedSignatureVerifier(Protocol):
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class BoundedAuthorization:
    approved_by: str
    expires_at: datetime
    policy_digest: str
    single_step_receipt_digest: str
    max_commands: int
    max_duration_seconds: float
    max_path_length_m: float
    max_error_count: int
    cycle_provider_digest: str
    digest: str
    signed_document: dict[str, object]


def _violation(code: RolloutCode, detail: str) -> RolloutViolation:
    return RolloutViolation(code, detail)


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"duplicate key: {key}")
        result[key] = value
    return result


def _load(path: Path, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except OSError as exc:
        raise _violation(RolloutCode.R_MISSING, f"{label} missing") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} invalid JSON") from exc
    if not isinstance(raw, dict):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be an object")
    return cast("dict[str, object]", raw)


def _text(doc: Mapping[str, object], key: str) -> str:
    value = doc.get(key)
    if not isinstance(value, str) or not value:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} invalid")
    return value


def _digest(doc: Mapping[str, object], key: str) -> str:
    value = _text(doc, key)
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise _violation(RolloutCode.R_HASH_MISMATCH, f"authorization {key} invalid")
    return value


def _time(doc: Mapping[str, object], key: str) -> datetime:
    try:
        result = datetime.fromisoformat(_text(doc, key).replace("Z", "+00:00"))
    except ValueError as exc:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} invalid") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"authorization {key} timezone")
    return result


def _positive(doc: Mapping[str, object], key: str) -> float:
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise _violation(RolloutCode.R_BUDGET_EXHAUSTED, f"authorization {key} invalid")
    return float(value)


def _verify_fixture(content: bytes, signature_hex: str) -> bool:
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    size = (_MODULUS.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    encoded = pow(int.from_bytes(signature, "big"), _EXPONENT, _MODULUS).to_bytes(size, "big")
    info = (
        bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(content).digest()
    )
    expected = b"\x00\x01" + b"\xff" * (size - len(info) - 3) + b"\x00" + info
    return hmac.compare_digest(encoded, expected)


def verify_single_step_receipt(path: Path) -> str:
    """Revalidate typed acknowledgement/newer post-state before promotion."""
    return verify_single_step_receipt_document(_load(path, "verified single-step receipt"))


def verify_bounded_authorization_document(
    doc: Mapping[str, object],
    *,
    now: datetime | None,
    single_step_receipt_digest: str,
    production_verifier: BoundedSignatureVerifier | None = None,
) -> BoundedAuthorization:
    """Authenticate signed authorization fields from persisted ledger bytes."""
    schema = _text(doc, "schema")
    scope = _text(doc, "artifact_scope")
    fixture_v1 = schema == _SCHEMA and frozenset(doc) == _V1_FIELDS
    production_v2 = schema == _PRODUCTION_SCHEMA and frozenset(doc) == _V2_FIELDS
    if not fixture_v1 and not production_v2:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization fields or schema")
    if scope == "production" and not production_v2:
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "production provider binding missing")
    content_doc = {k: v for k, v in doc.items() if k not in {"digest", "binding_signature"}}
    content = json.dumps(content_doc, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(content).hexdigest() != _digest(doc, "digest"):
        raise _violation(RolloutCode.R_HASH_MISMATCH, "authorization content digest")
    approved, expires = _time(doc, "approved_at"), _time(doc, "expires_at")
    if approved >= expires or (now is not None and (approved > now or now >= expires)):
        raise _violation(RolloutCode.R_STALE, "authorization is expired or not yet valid")
    signer, scheme = _text(doc, "signer_id"), _text(doc, "signature_scheme")
    signature = _text(doc, "binding_signature")
    fixture_ok = (
        scope == _SCOPE
        and signer == _SIGNER
        and scheme == _SCHEME
        and _verify_fixture(content, signature)
    )
    production_ok = (
        scope == "production"
        and production_verifier is not None
        and production_verifier.verify(signer, scheme, content, signature)
    )
    if not fixture_ok and not production_ok:
        raise _violation(RolloutCode.R_HASH_MISMATCH, "authorization signature invalid")
    if (
        signer != _text(doc, "approved_by")
        or _digest(doc, "single_step_receipt_digest") != single_step_receipt_digest
    ):
        raise _violation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization promotion binding")
    max_commands = int(_positive(doc, "max_commands"))
    max_errors = int(_positive(doc, "max_error_count"))
    if doc["max_commands"] != max_commands or doc["max_error_count"] != max_errors:
        raise _violation(RolloutCode.R_BUDGET_EXHAUSTED, "integer budget required")
    return BoundedAuthorization(
        signer,
        expires,
        _digest(doc, "policy_digest"),
        single_step_receipt_digest,
        max_commands,
        _positive(doc, "max_duration_seconds"),
        _positive(doc, "max_path_length_m"),
        max_errors,
        _digest(doc, "cycle_provider_digest") if production_v2 else "",
        _digest(doc, "digest"),
        dict(doc),
    )


def load_bounded_authorization(
    path: Path,
    *,
    now: datetime,
    single_step_receipt_digest: str,
    production_verifier: BoundedSignatureVerifier | None = None,
) -> BoundedAuthorization:
    """Load and authenticate one bounded authorization file."""
    return verify_bounded_authorization_document(
        _load(path, "authorization"),
        now=now,
        single_step_receipt_digest=single_step_receipt_digest,
        production_verifier=production_verifier,
    )
