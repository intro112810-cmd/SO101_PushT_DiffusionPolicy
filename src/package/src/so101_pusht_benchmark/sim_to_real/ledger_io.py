"""Strict JSONL file loading for hash-chained rollout ledgers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

from .ledger_chain import LedgerViolation
from .rollout_codes import RolloutCode

__all__ = ("LedgerDocument", "load_ledger_documents", "reject_duplicate_keys")
LedgerDocument: TypeAlias = dict[str, object]


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys at every nesting level."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"duplicate ledger key: {key}")
        result[key] = value
    return result


def load_ledger_documents(path: Path) -> list[LedgerDocument]:
    """Read every non-empty JSONL line as an object, rejecting malformed input."""
    documents: list[LedgerDocument] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "cannot read ledger") from exc
    for index, line in enumerate(lines):
        if not line.strip():
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"blank line {index + 1}")
        try:
            parsed = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise LedgerViolation(
                RolloutCode.R_HASH_MISMATCH, f"invalid JSON line {index + 1}"
            ) from exc
        if not isinstance(parsed, dict):
            raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, f"line {index + 1} not an object")
        documents.append(cast("LedgerDocument", parsed))
    if not documents:
        raise LedgerViolation(RolloutCode.R_HASH_MISMATCH, "empty ledger")
    return documents
