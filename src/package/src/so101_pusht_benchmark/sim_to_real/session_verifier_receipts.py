"""Cross-receipt semantics for guarded-rollout session verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from .ledger_chain import LedgerRecord, parse_ledger_records, replay_digest
from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "verify_bounded",
    "verify_cleanup",
    "verify_physical_ledger",
    "verify_shadow",
    "verify_single",
)

_CLEANUP_SCHEMA = "guarded-rollout-cleanup-v1"
_REQUIRED_LEDGER_KINDS = (
    "lineage",
    "policy",
    "samples",
    "joint",
    "camera",
    "cartesian_proposal",
    "ik_proposal",
    "supervisor_decision",
    "intent",
    "dispatch_status",
    "ack",
    "post_state",
    "cleanup",
)


def _fail(detail: str) -> None:
    raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, detail)


def _string(record: LedgerRecord, field: str, label: str) -> str:
    value = record.content.get(field)
    if not isinstance(value, str) or not value:
        _fail(f"{label} {field} is missing")
    return cast("str", value)


def _number(record: LedgerRecord, field: str, label: str) -> float:
    value = record.content.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} {field} is not numeric")
    result = float(cast("int | float", value))
    if not math.isfinite(result):
        _fail(f"{label} {field} is nonfinite")
    return result


def _float_row(record: LedgerRecord, field: str, label: str) -> tuple[float, ...]:
    value = record.content.get(field)
    if not isinstance(value, list):
        _fail(f"{label} {field} is missing")
    values = cast("list[object]", value)
    if len(values) != 5:
        _fail(f"{label} {field} must contain five body joints")
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail(f"{label} {field} contains a non-number")
        number = float(cast("int | float", item))
        if not math.isfinite(number):
            _fail(f"{label} {field} contains a nonfinite value")
        result.append(number)
    return tuple(result)


def verify_physical_ledger(
    records_raw: Sequence[Mapping[str, object]],
    *,
    lineage_digest: str,
    joint_digest: str,
    camera_digest: str,
    single_receipt: Mapping[str, object],
) -> tuple[str, str, int]:
    """Authenticate event order and bind intent, acknowledgement, and post-state."""
    records = parse_ledger_records(records_raw)
    if tuple(record.kind for record in records) != _REQUIRED_LEDGER_KINDS:
        _fail("physical ledger event sequence is incomplete")
    by_kind = {record.kind: record for record in records}
    if _string(by_kind["lineage"], "lineage_digest", "ledger") != lineage_digest:
        _fail("physical ledger lineage binding mismatch")
    if _string(by_kind["joint"], "joint_digest", "ledger") != joint_digest:
        _fail("physical ledger joint binding mismatch")
    if _string(by_kind["camera"], "camera_digest", "ledger") != camera_digest:
        _fail("physical ledger camera binding mismatch")
    intent = by_kind["intent"]
    acknowledgement = by_kind["ack"]
    dispatch = by_kind["dispatch_status"]
    post_state = by_kind["post_state"]
    command_id = _string(intent, "command_id", "ledger intent")
    bound = (dispatch, acknowledgement, post_state, by_kind["cleanup"])
    if any(
        _string(record, "command_id", f"ledger {record.kind}") != command_id for record in bound
    ):
        _fail("physical ledger command binding mismatch")
    if _string(dispatch, "status", "ledger dispatch") != "acknowledged":
        _fail("physical ledger dispatch is not acknowledged")
    if _number(dispatch, "write_count", "ledger dispatch") != 1.0:
        _fail("physical ledger dispatch count is not one")
    if _float_row(intent, "body_degrees", "ledger intent") != _float_row(
        acknowledgement, "accepted_body_degrees", "ledger acknowledgement"
    ):
        _fail("physical ledger acknowledgement differs from intent")
    if _number(post_state, "post_state_created_at", "ledger post-state") <= _number(
        post_state, "previous_sample_created_at", "ledger post-state"
    ):
        _fail("physical ledger post-state is not newer")
    if _string(by_kind["cleanup"], "status", "ledger cleanup") != "released":
        _fail("physical ledger cleanup is incomplete")
    if single_receipt.get("command_id") != command_id:
        _fail("single-step receipt command does not match ledger")
    return records[-1].digest, replay_digest(records_raw), len(records)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return cast("int", value)


def verify_shadow(
    receipt: Mapping[str, object], records_raw: Sequence[Mapping[str, object]]
) -> int:
    """Require a complete, capability-free shadow campaign and matching ledger."""
    records = parse_ledger_records(records_raw)
    if receipt.get("mode") != "sim_to_real_continuous_shadow_campaign":
        _fail("shadow receipt mode mismatch")
    if receipt.get("terminal_state") != "SHADOW_COMPLETE":
        _fail("shadow campaign is not complete")
    cycles = _positive_int(receipt.get("cycles_completed"), "shadow cycles")
    if cycles < 1 or receipt.get("cycles_completed") != receipt.get("cycle_limit"):
        _fail("shadow campaign budget is incomplete")
    if receipt.get("motor_writes_performed") is not False:
        _fail("shadow receipt reports a motor write")
    if receipt.get("actuation_performed") is not False or receipt.get("writer_symbols") != 0:
        _fail("shadow receipt is not capability-free")
    if receipt.get("ledger_digest") != records[-1].digest:
        _fail("shadow ledger digest mismatch")
    return cycles


def verify_single(receipt: Mapping[str, object]) -> int:
    """Require exactly one acknowledged complete commissioning command."""
    count = _positive_int(receipt.get("write_count"), "single-step write_count")
    if receipt.get("state") != "COMPLETE" or count != 1:
        _fail("single-step receipt is not one verified complete command")
    if receipt.get("motor_writes_performed") is not True:
        _fail("single-step receipt does not report its fixture command")
    return count


def verify_bounded(receipt: Mapping[str, object]) -> int:
    """Require a complete bounded receipt with unique commands inside budget."""
    count = _positive_int(receipt.get("write_count"), "bounded write_count")
    budget = _positive_int(receipt.get("max_commands"), "bounded max_commands")
    raw_ids = receipt.get("command_ids")
    if not isinstance(raw_ids, list):
        _fail("bounded command inventory is malformed")
    command_ids: list[str] = []
    for item in cast("list[object]", raw_ids):
        if not isinstance(item, str) or not item:
            _fail("bounded command inventory is malformed")
        command_ids.append(cast("str", item))
    consistent = (
        receipt.get("mode") == "sim_to_real_bounded_rollout"
        and receipt.get("state") == "COMPLETE"
        and receipt.get("fault_code") is None
        and count == len(command_ids)
        and count <= budget
        and len(set(command_ids)) == len(command_ids)
        and receipt.get("motor_writes_performed") is (count > 0)
    )
    if not consistent:
        _fail("bounded rollout receipt is inconsistent")
    return count


def verify_cleanup(receipt: Mapping[str, object]) -> None:
    """Require an explicit empty resource inventory and closed devices."""
    if receipt.get("schema") != _CLEANUP_SCHEMA or receipt.get("all_released") is not True:
        _fail("cleanup receipt is not complete")
    empty_fields = (
        "task_owned_process_ids_remaining",
        "open_device_fds",
        "temporary_paths_remaining",
        "staging_paths_remaining",
    )
    if any(receipt.get(field) != [] for field in empty_fields):
        _fail("cleanup receipt reports remaining resources")
    open_fields = ("writer_open", "camera_open", "serial_open")
    if any(receipt.get(field) is not False for field in open_fields):
        _fail("cleanup receipt reports an open device")
