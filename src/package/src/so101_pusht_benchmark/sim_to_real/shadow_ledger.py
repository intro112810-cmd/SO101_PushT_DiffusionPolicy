"""Non-actuating hash-chained ledger append helpers for shadow campaigns."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from so101_pusht_benchmark.sim_to_real.ledger_chain import GENESIS_DIGEST, canonical_hash
from so101_pusht_benchmark.sim_to_real.ledger_io import LedgerDocument
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    locate_receipt_path,
    validate_receipt_identity,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.secure_io import atomic_write_new, unlink_owned_leaf
from so101_pusht_benchmark.sim_to_real.shadow_types import ShadowCampaignResult

__all__ = (
    "LedgerDocument",
    "append_record",
    "persist_campaign",
    "serialize_ledger",
    "write_shadow_receipt",
)


def _appended_record(
    content: LedgerDocument,
    *,
    previous_digest: str,
    sequence: int,
) -> LedgerDocument:
    record = {**content, "prev_digest": previous_digest, "sequence": sequence}
    return {**record, "digest": canonical_hash(record)}


def _record_digest(record: LedgerDocument) -> str:
    digest = record["digest"]
    if not isinstance(digest, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "ledger digest must be a string")
    return digest


def append_record(
    records: list[LedgerDocument],
    content: LedgerDocument,
    *,
    previous_digest: str,
) -> str:
    """Append one hash-chained non-actuating ledger record and return its digest."""
    record = _appended_record(
        content,
        previous_digest=previous_digest,
        sequence=len(records),
    )
    records.append(record)
    return _record_digest(record)


def serialize_ledger(records: Sequence[LedgerDocument]) -> str:
    """Serialize one verified hash-chained ledger for a durable write."""
    if not records or _record_digest(records[0]) == GENESIS_DIGEST:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "empty ledger")
    return "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )


def persist_campaign(
    records: Sequence[LedgerDocument], result: ShadowCampaignResult, output_dir: Path
) -> None:
    """Publish a campaign ledger and receipt, cleaning only the owned ledger on failure."""
    location = locate_receipt_path(output_dir)
    ledger_identity = atomic_write_new(
        location.resolved,
        result.ledger_path.name,
        serialize_ledger(records).encode("utf-8"),
        temporary=".ledger.jsonl.tmp",
    )
    accepted = False
    try:
        validate_receipt_identity(location, production=result.evidence_scope == "production")
        write_shadow_receipt(result, output_dir)
        accepted = True
    finally:
        if not accepted:
            unlink_owned_leaf(ledger_identity)


def write_shadow_receipt(result: ShadowCampaignResult, output_dir: Path) -> Path:
    """Atomically persist the unique terminal shadow receipt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = (
        output_dir / "SHADOW_COMPLETE"
        if result.terminal_state == "SHADOW_COMPLETE"
        else output_dir / "terminal_receipt.json"
    )
    encoded = (json.dumps(result.to_document(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    location = locate_receipt_path(target)
    identity = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        encoded,
        temporary=f".{target.name}.tmp",
    )
    accepted = False
    try:
        validate_receipt_identity(location, production=result.evidence_scope == "production")
        accepted = True
        return location.lexical
    finally:
        if not accepted:
            unlink_owned_leaf(identity)
