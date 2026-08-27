"""Safe, read-only loading for guarded-rollout session evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import cast, NoReturn

from .receipt_routing import (
    locate_receipt_path,
    ReceiptRoutingError,
    validate_receipt_identity,
)
from .replay_receipts import require_digest
from .rollout_codes import RolloutCode, RolloutViolation
from .session_manifest import (
    CANONICAL_SESSION_MEMBERS,
    SESSION_MANIFEST_SCHEMA,
    manifest_digest,
)
from .session_scope import derive_session_evidence_scope

__all__ = ("LoadedSession", "load_json", "load_ledger", "load_session")

_SCHEMA = "guarded-rollout-session-v1"
_REQUIRED_MEMBERS = frozenset(
    {
        "bounded_receipt.json",
        "camera-registration.json",
        "cleanup.json",
        "joint-equivalence.json",
        "lineage.json",
        "physical_ledger.jsonl",
        "shadow_ledger.jsonl",
        "shadow_receipt.json",
        "single_step_receipt.json",
    }
)


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """Authenticated session manifest and its exact regular-file members."""

    session_id: str
    evidence_scope: str
    lineage_authority_digest: str
    manifest_digest: str
    lexical_root: Path
    resolved_root: Path
    members: dict[str, Path]


def _fail(detail: str, code: RolloutCode = RolloutCode.R_HASH_MISMATCH) -> NoReturn:
    raise RolloutViolation(code, detail)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON mapping", RolloutCode.R_MISSING)
    return cast("dict[str, object]", value)


def load_json(path: Path, label: str) -> dict[str, object]:
    """Read one JSON mapping without changing it."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {label}") from exc
    return _mapping(value, label)


def load_ledger(path: Path, label: str) -> list[dict[str, object]]:
    """Read one JSON-lines ledger into boundary-typed mappings."""
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {label}") from exc
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutViolation(
                RolloutCode.R_HASH_MISMATCH, f"{label} record {index} is malformed"
            ) from exc
        records.append(_mapping(value, f"{label} record {index}"))
    return records


def _session_root(resolved: Path) -> Path:
    if resolved.is_symlink() or not resolved.is_dir():
        _fail("session must be a real directory", RolloutCode.R_MISSING)
    root = resolved
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            _fail(f"session contains a non-regular member: {path.name}")
    return root


def _member_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.name != relative:
        _fail(f"unsafe session member: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        _fail(f"missing session member: {relative}", RolloutCode.R_MISSING)
    return path


def _legacy_members(root: Path, manifest: Mapping[str, object]) -> dict[str, Path]:
    raw_files = _mapping(manifest.get("files"), "session files")
    if frozenset(raw_files) != _REQUIRED_MEMBERS:
        _fail("session membership is incomplete or contains extras")
    actual = frozenset(
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.name != "session_manifest.json"
    )
    if actual != _REQUIRED_MEMBERS:
        _fail("session directory membership does not match manifest")
    members: dict[str, Path] = {}
    for relative, expected in raw_files.items():
        path = _member_path(root, relative)
        declared = require_digest(expected, f"session member {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != declared:
            _fail(f"session member digest mismatch: {relative}")
        members[relative] = path
    return members


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _canonical_members(root: Path, manifest: Mapping[str, object]) -> dict[str, Path]:
    raw_files = _mapping(manifest.get("files"), "session files")
    for relative in raw_files:
        _member_path(root, relative)
    if frozenset(raw_files) != CANONICAL_SESSION_MEMBERS:
        _fail("canonical session membership is incomplete or contains extras")
    actual = frozenset(
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.name != "session_manifest.json"
    )
    if actual != CANONICAL_SESSION_MEMBERS:
        _fail("session directory membership does not match manifest")
    members: dict[str, Path] = {}
    for relative, raw_record in raw_files.items():
        record = _mapping(raw_record, f"session member record {relative}")
        if set(record) != {"sha256", "size", "mtime_ns"}:
            _fail(f"session member record is incomplete: {relative}")
        path = _member_path(root, relative)
        info = path.stat()
        declared = require_digest(record.get("sha256"), f"session member {relative}")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != declared
            or info.st_size != _integer(record.get("size"), f"session member size {relative}")
            or info.st_mtime_ns
            != _integer(record.get("mtime_ns"), f"session member mtime {relative}")
        ):
            _fail(f"session member is tampered or stale: {relative}")
        members[relative] = path
    return members


def load_session(session: Path) -> LoadedSession:
    """Authenticate path identity, exact members, and member-derived scope."""
    try:
        location = locate_receipt_path(session)
    except ReceiptRoutingError as exc:
        _fail(str(exc), RolloutCode.R_MISSING)
    root = _session_root(location.resolved)
    manifest_path = root / "session_manifest.json"
    manifest = load_json(manifest_path, "session manifest")
    schema = manifest.get("schema")
    raw_session_id = manifest.get("session_id")
    if not isinstance(raw_session_id, str) or not raw_session_id:
        _fail("session id is missing", RolloutCode.R_MISSING)
    raw_scope = manifest.get("evidence_scope")
    if schema == _SCHEMA:
        if raw_scope != "fixture_non_actuating_qa":
            _fail("legacy fixture manifest cannot represent production evidence")
        try:
            validate_receipt_identity(location, production=False)
        except ReceiptRoutingError as exc:
            _fail(str(exc), RolloutCode.R_MISSING)
        authority = require_digest(manifest.get("lineage_authority_digest"), "lineage authority")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        members = _legacy_members(root, manifest)
        evidence_scope = "fixture_non_actuating_qa"
    elif schema == SESSION_MANIFEST_SCHEMA:
        if manifest.get("status") != "COMPLETE":
            _fail("session manifest is partial")
        for relative in _mapping(manifest.get("files"), "session files"):
            _member_path(root, relative)
        declared = require_digest(manifest.get("session_digest"), "session digest")
        if manifest_digest(dict(manifest)) != declared:
            _fail("session manifest digest mismatch")
        members = _canonical_members(root, manifest)
        derived_scope = derive_session_evidence_scope(root)
        if raw_scope != derived_scope:
            _fail("manifest evidence scope differs from derived evidence scope")
        roots = _mapping(manifest.get("session_root"), "session root identity")
        if set(roots) != {"lexical", "resolved"} or roots != {
            "lexical": str(location.lexical),
            "resolved": str(location.resolved),
        }:
            _fail("session lexical/resolved root identity mismatch")
        try:
            validate_receipt_identity(
                location,
                production=derived_scope == "authorized_physical_diagnostic",
            )
        except ReceiptRoutingError as exc:
            _fail(str(exc), RolloutCode.R_MISSING)
        lineage = load_json(root / "lineage.json", "lineage receipt")
        authority = require_digest(lineage.get("authority_digest"), "lineage authority")
        digest = declared
        evidence_scope = derived_scope
    else:
        _fail("unsupported session schema")
    return LoadedSession(
        session_id=raw_session_id,
        evidence_scope=evidence_scope,
        lineage_authority_digest=authority,
        manifest_digest=digest,
        lexical_root=location.lexical,
        resolved_root=location.resolved,
        members=members,
    )
