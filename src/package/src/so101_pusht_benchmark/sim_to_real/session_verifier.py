"""Read-only verification for a complete guarded sim-to-real session."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from .replay_receipts import (
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from .session_verifier_io import load_json, load_ledger, load_session
from .session_verifier_receipts import (
    verify_bounded,
    verify_cleanup as verify_cleanup_receipt,
    verify_physical_ledger,
    verify_shadow,
    verify_single,
)

__all__ = ("SessionVerificationReceipt", "verify_guarded_session")


@dataclass(frozen=True, slots=True)
class SessionVerificationReceipt:
    """Stable non-actuating result emitted only after every gate passes."""

    valid: bool
    session_id: str
    evidence_scope: str
    session_digest: str
    ledger_terminal_digest: str
    ledger_replay_digest: str
    ledger_record_count: int
    shadow_cycles: int
    single_step_write_count: int
    bounded_write_count: int
    cleanup_verified: bool
    actuation_performed_by_verifier: bool = False

    def to_document(self) -> dict[str, object]:
        """Return the machine-consumed JSON representation."""
        return asdict(self)


def verify_guarded_session(session: Path, *, verify_cleanup: bool) -> SessionVerificationReceipt:
    """Authenticate a completed session without opening devices or changing evidence."""
    loaded = load_session(session)
    members = loaded.members
    lineage = load_json(members["lineage.json"], "lineage receipt")
    validated_lineage = validate_lineage_receipt(
        lineage,
        expected_digest=loaded.lineage_authority_digest,
    )
    lineage_digest = cast("str", validated_lineage["lineage_digest"])
    joint_digest = validate_joint_receipt(
        load_json(members["joint-equivalence.json"], "joint receipt")
    )
    camera_digest = validate_camera_receipt(
        load_json(members["camera-registration.json"], "camera receipt")
    )
    single = load_json(members["single_step_receipt.json"], "single-step receipt")
    terminal, replay, record_count = verify_physical_ledger(
        load_ledger(members["physical_ledger.jsonl"], "physical ledger"),
        lineage_digest=lineage_digest,
        joint_digest=joint_digest,
        camera_digest=camera_digest,
        single_receipt=single,
    )
    shadow_cycles = verify_shadow(
        load_json(members["shadow_receipt.json"], "shadow receipt"),
        load_ledger(members["shadow_ledger.jsonl"], "shadow ledger"),
    )
    single_count = verify_single(single)
    bounded_count = verify_bounded(load_json(members["bounded_receipt.json"], "bounded receipt"))
    if verify_cleanup:
        verify_cleanup_receipt(load_json(members["cleanup.json"], "cleanup receipt"))
    return SessionVerificationReceipt(
        valid=True,
        session_id=loaded.session_id,
        evidence_scope=loaded.evidence_scope,
        session_digest=loaded.manifest_digest,
        ledger_terminal_digest=terminal,
        ledger_replay_digest=replay,
        ledger_record_count=record_count,
        shadow_cycles=shadow_cycles,
        single_step_write_count=single_count,
        bounded_write_count=bounded_count,
        cleanup_verified=verify_cleanup,
    )
