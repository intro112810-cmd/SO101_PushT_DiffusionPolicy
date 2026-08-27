"""Build the one exact, content-addressed guarded-session manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Final

from .ledger_io import LedgerDocument
from .receipt_routing import (
    locate_receipt_path,
    validate_receipt_identity,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .secure_io import atomic_write_new, read_regular_leaf, unlink_owned_leaf
from .session_scope import derive_session_evidence_scope

__all__ = (
    "CANONICAL_SESSION_MEMBERS",
    "SESSION_MANIFEST_SCHEMA",
    "manifest_digest",
    "write_session_manifest",
)

SESSION_MANIFEST_SCHEMA: Final = "guarded-rollout-session-v2"
CANONICAL_SESSION_MEMBERS: Final = frozenset(
    {
        "approved_policy.yaml",
        "arming_receipt.json",
        "bounded_acknowledgements.jsonl",
        "bounded_authorization.json",
        "bounded_intents.jsonl",
        "bounded_post_states.jsonl",
        "bounded_receipt.json",
        "camera-registration.json",
        "cleanup.json",
        "fault.json",
        "inference_replay.json",
        "joint-equivalence.json",
        "lineage.json",
        "physical_ledger.jsonl",
        "raw_samples.json",
        "shadow_ledger.jsonl",
        "shadow_receipt.json",
        "single_step_acknowledgement.json",
        "single_step_authorization.json",
        "single_step_intent.json",
        "single_step_post_state.json",
        "single_step_receipt.json",
    }
)


def _canonical_bytes(value: LedgerDocument) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def manifest_digest(document: LedgerDocument) -> str:
    """Hash every manifest field except the digest that carries this hash."""
    content = {key: value for key, value in document.items() if key != "session_digest"}
    return hashlib.sha256(_canonical_bytes(content)).hexdigest()


def _member_record(path: Path) -> LedgerDocument:
    try:
        content, info = read_regular_leaf(path.parent, path.name)
    except RolloutViolation as exc:
        raise RolloutViolation(
            RolloutCode.R_HASH_MISMATCH, f"unsafe session member: {path.name}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"unsafe session member: {path.name}")
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def write_session_manifest(session: Path, *, session_id: str) -> Path:
    """Derive scope and finalize one complete session without trusting caller claims."""
    if not session_id or "/" in session_id or ".." in session_id:
        raise RolloutViolation(RolloutCode.R_MISSING, "unsafe session id")
    location = locate_receipt_path(session)
    root = location.resolved
    if not root.is_dir():
        raise RolloutViolation(RolloutCode.R_MISSING, "session staging directory is missing")
    manifest_io = root / "session_manifest.json"
    manifest_lexical = location.lexical / "session_manifest.json"
    if manifest_io.exists() or manifest_io.is_symlink():
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "session manifest already exists")
    actual = frozenset(path.name for path in root.iterdir())
    if actual != CANONICAL_SESSION_MEMBERS:
        raise RolloutViolation(
            RolloutCode.R_MISSING,
            "canonical session membership is incomplete or contains extras",
        )
    evidence_scope = derive_session_evidence_scope(root)
    production = evidence_scope == "authorized_physical_diagnostic"
    validate_receipt_identity(location, production=production)
    files = {name: _member_record(root / name) for name in sorted(CANONICAL_SESSION_MEMBERS)}
    document: LedgerDocument = {
        "schema": SESSION_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "session_id": session_id,
        "evidence_scope": evidence_scope,
        "session_root": {
            "lexical": str(location.lexical),
            "resolved": str(location.resolved),
        },
        "files": files,
    }
    document["session_digest"] = manifest_digest(document)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    identity = atomic_write_new(
        root,
        manifest_io.name,
        encoded,
        temporary=".session_manifest.json.partial",
    )
    accepted = False
    try:
        validate_receipt_identity(location, production=production)
        accepted = True
        return manifest_lexical
    finally:
        if not accepted:
            unlink_owned_leaf(identity)
