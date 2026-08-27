"""Append-only, hash-chained session ledger and deterministic replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import string
from typing import TypeAlias, cast

from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "GENESIS_DIGEST",
    "LedgerDigest",
    "LedgerRecord",
    "LedgerViolation",
    "canonical_hash",
    "parse_ledger_records",
    "replay_digest",
    "verify_ledger",
)
GENESIS_DIGEST = "0" * 64
_REQUIRED_FIELDS = frozenset({"kind", "prev_digest", "digest", "sequence"})
_DISPATCH_KINDS = frozenset({"dispatch", "dispatch_status"})
_HEX = frozenset(string.hexdigits.lower())
LedgerValue: TypeAlias = (
    str | int | float | bool | list["LedgerValue"] | dict[str, "LedgerValue"] | None
)


class LedgerViolation(RolloutViolation):
    """A ledger chain break; it never carries physical writer capability."""


@dataclass(frozen=True, slots=True)
class LedgerDigest:
    """One canonical 64-hex ledger digest."""

    value: str

    def __post_init__(self) -> None:
        """Reject any digest that is not exactly 64 lowercase hex characters."""
        if len(self.value) != 64 or any(character not in _HEX for character in self.value):
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "ledger digest")


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """One validated ledger entry carrying its authenticated chain digest."""

    kind: str
    prev_digest: str
    digest: str
    content: dict[str, LedgerValue]


def canonical_hash(mapping: Mapping[str, object]) -> str:
    """SHA-256 of canonical JSON with sorted keys and no whitespace."""
    encoded = json.dumps(
        mapping,
        allow_nan=True,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"{label} is not a string")
    return LedgerDigest(value).value


def _content(raw: Mapping[str, object]) -> dict[str, LedgerValue]:
    result: dict[str, LedgerValue] = {}
    for key, value in raw.items():
        result[key] = _value(value, key)
    return result


def _value(value: object, key: str) -> LedgerValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, list):
        return [_value(item, key) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        inner: dict[str, LedgerValue] = {}
        for inner_key, inner_value in cast("dict[object, object]", value).items():
            if not isinstance(inner_key, str):
                raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"non-string key in {key}")
            inner[inner_key] = _value(inner_value, inner_key)
        return inner
    raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"unsupported field {key}")


def parse_ledger_records(raw_records: Sequence[Mapping[str, object]]) -> list[LedgerRecord]:
    """Parse and authenticate one full ledger, fail closed on any break."""
    records: list[LedgerRecord] = []
    previous_digest = GENESIS_DIGEST
    for index, raw in enumerate(raw_records):
        missing = _REQUIRED_FIELDS - frozenset(raw)
        if missing:
            raise LedgerViolation(
                RolloutCode.R_HASH_MISMATCH, f"record {index} missing {sorted(missing)}"
            )
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"record {index} kind")
        sequence = raw.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != index:
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"record {index} sequence")
        prev_digest = _sha(raw.get("prev_digest"), f"record {index} prev_digest")
        digest = _sha(raw.get("digest"), f"record {index} digest")
        if prev_digest != previous_digest:
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"record {index} chain link")
        content = _content({key: value for key, value in raw.items() if key != "digest"})
        if canonical_hash(content) != digest:
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"record {index} content")
        records.append(
            LedgerRecord(kind=kind, prev_digest=prev_digest, digest=digest, content=content)
        )
        previous_digest = digest
    if not records:
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "empty ledger")
    if records[-1].kind != "cleanup":
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "ledger missing terminal cleanup")
    intent_seen = any(record.kind == "intent" for record in records)
    if not intent_seen and any(record.kind in _DISPATCH_KINDS for record in records):
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "dispatch without prior intent")
    return records


def verify_ledger(records: Sequence[Mapping[str, object]]) -> str:
    """Return the terminal digest only when the whole chain authenticates."""
    return parse_ledger_records(records)[-1].digest


def replay_digest(records: Sequence[Mapping[str, object]]) -> str:
    """Concatenate authenticated record digests and hash the stream."""
    parsed = parse_ledger_records(records)
    combined = hashlib.sha256()
    for record in parsed:
        combined.update(record.digest.encode("ascii"))
    return combined.hexdigest()
