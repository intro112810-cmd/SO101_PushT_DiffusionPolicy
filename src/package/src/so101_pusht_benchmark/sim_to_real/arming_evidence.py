"""Independent content-addressed operational evidence for single-step arming."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import cast

from .ledger_chain import canonical_hash
from .rollout_codes import RolloutCode, RolloutViolation

__all__ = ("OperationalEvidence", "load_operational_evidence")


@dataclass(frozen=True, slots=True)
class OperationalEvidence:
    """Fresh device ownership, physical interlock, and existing torque receipts."""

    ownership_digest: str
    interlock_digest: str
    torque_digest: str


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} evidence missing")
    return cast("dict[str, object]", value)


def _load(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} evidence missing") from exc
    return _mapping(value, label)


def _text(document: Mapping[str, object], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} {key} missing")
    return value


def _boolean(document: Mapping[str, object], key: str, label: str) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} {key} missing")
    return value


def _timestamp(document: Mapping[str, object], label: str) -> datetime:
    try:
        value = datetime.fromisoformat(_text(document, "observed_at", label).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RolloutViolation(RolloutCode.R_STALE, f"{label} timestamp invalid") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise RolloutViolation(RolloutCode.R_STALE, f"{label} timestamp lacks timezone")
    return value


def _validate_digest(document: Mapping[str, object], label: str) -> str:
    digest = _text(document, "digest", label)
    content = {key: value for key, value in document.items() if key != "digest"}
    if canonical_hash(content) != digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"{label} digest mismatch")
    return digest


def _fresh(document: Mapping[str, object], label: str, now: datetime, max_age: float) -> None:
    observed_at = _timestamp(document, label)
    if observed_at > now or (now - observed_at).total_seconds() > max_age:
        raise RolloutViolation(RolloutCode.R_STALE, f"{label} evidence stale")


def load_operational_evidence(
    directory: Path,
    *,
    now: datetime,
    max_age_seconds: float,
) -> OperationalEvidence:
    """Load three independent receipts; no unavailable live read is inferred as green."""
    ownership = _load(directory / "device_ownership.json", "device ownership")
    interlock = _load(directory / "interlock.json", "interlock")
    torque = _load(directory / "torque_state.json", "torque state")
    if (
        set(ownership)
        != {
            "kind",
            "observed_at",
            "serial_device",
            "camera_device",
            "exclusive_owner",
            "competing_holder",
            "digest",
        }
        or ownership["kind"] != "device_ownership"
    ):
        raise RolloutViolation(RolloutCode.R_OWNERSHIP_CONFLICT, "ownership fields invalid")
    if (
        set(interlock) != {"kind", "observed_at", "deadman_active", "stop_clear", "digest"}
        or interlock["kind"] != "physical_interlock"
    ):
        raise RolloutViolation(RolloutCode.R_DEADMAN_INACTIVE, "interlock fields invalid")
    if (
        set(torque) != {"kind", "observed_at", "state", "read_method", "digest"}
        or torque["kind"] != "existing_torque_state"
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "torque fields invalid")
    for document, label in (
        (ownership, "device ownership"),
        (interlock, "interlock"),
        (torque, "torque state"),
    ):
        _fresh(document, label, now, max_age_seconds)
    if _boolean(ownership, "competing_holder", "ownership") or not _boolean(
        ownership, "exclusive_owner", "ownership"
    ):
        raise RolloutViolation(RolloutCode.R_OWNERSHIP_CONFLICT, "device ownership conflicted")
    if not _boolean(interlock, "deadman_active", "interlock") or not _boolean(
        interlock, "stop_clear", "interlock"
    ):
        raise RolloutViolation(RolloutCode.R_DEADMAN_INACTIVE, "deadman or stop inactive")
    if (
        _text(torque, "state", "torque") != "unmodified"
        or _text(torque, "read_method", "torque") != "approved_read_only_receipt"
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "torque state unapproved")
    return OperationalEvidence(
        _validate_digest(ownership, "device ownership"),
        _validate_digest(interlock, "interlock"),
        _validate_digest(torque, "torque state"),
    )
