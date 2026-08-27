"""Derive session evidence scope from authenticated members, never caller claims."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import cast, Final, Literal

import yaml

from .bounded_authorization import verify_bounded_authorization_document
from .policy_parser import load_fixture_safety_policy
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import load_single_step_authorization

__all__ = ("EvidenceScope", "derive_session_evidence_scope")

EvidenceScope = Literal["test_fixture_only", "authorized_physical_diagnostic"]
_FIXTURE_VALUES: Final = {
    "test_fixture_only",
    "fixture_deterministic_adapter",
    "fixture_fake_bus_not_production",
    "fixture_adapter_not_frozen_production",
    "fixture_non_actuating_qa",
}


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} must be a mapping")
    return cast("dict[str, object]", value)


def _json(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {label}") from exc
    return _mapping(value, label)


def _fixture_value(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return (
            lowered in _FIXTURE_VALUES
            or lowered.startswith(("fixture-", "fixture_"))
            or "fake_bus" in lowered
            or "fixturebus" in lowered
        )
    if isinstance(value, list):
        return any(_fixture_value(item) for item in cast("list[object]", value))
    if isinstance(value, dict):
        return any(_fixture_value(item) for item in cast("dict[object, object]", value).values())
    return False


def _jsonl_has_fixture_marker(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {path.name}") from exc
    if not lines:
        raise RolloutViolation(RolloutCode.R_MISSING, f"empty session member: {path.name}")
    for line in lines:
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RolloutViolation(
                RolloutCode.R_HASH_MISMATCH, f"malformed session member: {path.name}"
            ) from exc
        if _fixture_value(value):
            return True
    return False


def _fixture_policy(path: Path) -> bool:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, "cannot read approved policy") from exc
    document = _mapping(raw, "approved policy")
    if document.get("artifact_scope") != "test_fixture_only":
        return False
    approved = document.get("approved_at")
    if not isinstance(approved, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "policy approved_at missing")
    now = datetime.fromisoformat(approved.replace("Z", "+00:00")) + timedelta(seconds=1)
    load_fixture_safety_policy(path, now=now)
    return True


def _fixture_single_authorization(path: Path) -> bool:
    document = _json(path, "single-step authorization")
    if document.get("artifact_scope") != "test_fixture_only":
        return False
    approved = document.get("approved_at")
    if not isinstance(approved, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "authorization approved_at missing")
    now = datetime.fromisoformat(approved.replace("Z", "+00:00")) + timedelta(seconds=1)
    loaded = load_single_step_authorization(path, now=now)
    return loaded.signer_id == loaded.approved_by and "test-fixture" in loaded.signature_scheme


def _fixture_bounded_authorization(path: Path) -> bool:
    document = _json(path, "bounded authorization")
    if document.get("artifact_scope") != "test_fixture_only":
        return False
    approved = document.get("approved_at")
    single_digest = document.get("single_step_receipt_digest")
    if not isinstance(approved, str) or not isinstance(single_digest, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "bounded authorization binding missing")
    now = datetime.fromisoformat(approved.replace("Z", "+00:00")) + timedelta(seconds=1)
    loaded = verify_bounded_authorization_document(
        document,
        now=now,
        single_step_receipt_digest=single_digest,
    )
    return loaded.signed_document.get("artifact_scope") == "test_fixture_only"


def derive_session_evidence_scope(root: Path) -> EvidenceScope:
    """Authenticate fixture authorities or fail closed without production providers."""
    lineage = _json(root / "lineage.json", "lineage receipt")
    artifact_id = lineage.get("artifact_id")
    fixture_lineage = (
        isinstance(artifact_id, str) and artifact_id.startswith("fixture-")
    ) or lineage.get("fixture_only") is True
    fixture_policy = _fixture_policy(root / "approved_policy.yaml")
    fixture_single = _fixture_single_authorization(root / "single_step_authorization.json")
    fixture_bounded = _fixture_bounded_authorization(root / "bounded_authorization.json")
    fixture_receipt = any(
        _fixture_value(_json(root / name, name))
        for name in (
            "arming_receipt.json",
            "bounded_receipt.json",
            "fault.json",
            "inference_replay.json",
            "raw_samples.json",
            "single_step_acknowledgement.json",
            "single_step_intent.json",
            "single_step_post_state.json",
            "single_step_receipt.json",
        )
    )
    fixture_ledger = any(
        _jsonl_has_fixture_marker(root / name)
        for name in (
            "bounded_acknowledgements.jsonl",
            "bounded_intents.jsonl",
            "bounded_post_states.jsonl",
            "physical_ledger.jsonl",
            "shadow_ledger.jsonl",
        )
    )
    if any(
        (
            fixture_lineage,
            fixture_policy,
            fixture_single,
            fixture_bounded,
            fixture_receipt,
            fixture_ledger,
        )
    ):
        return "test_fixture_only"
    raise RolloutViolation(
        RolloutCode.R_POLICY_UNAUTHORIZED,
        "governed production policy, authorization verifiers, and providers are unavailable",
    )
