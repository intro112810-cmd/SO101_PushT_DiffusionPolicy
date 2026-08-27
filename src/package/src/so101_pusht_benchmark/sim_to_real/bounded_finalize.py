"""Terminal ledger and receipt finalization for bounded rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .bounded_authorization import BoundedAuthorization
from .bounded_types import BoundedRolloutResult
from .shadow_ledger import append_record, serialize_ledger
from .single_step import FixtureBus, count_writes


@dataclass(frozen=True, slots=True)
class BoundedFinalization:
    records: list[dict[str, object]]
    previous: str
    fault: str | None
    fault_detail: str | None
    error_count: int
    authorization: BoundedAuthorization
    bus: FixtureBus
    command_ids: list[str]
    output: Path


def finalize_bounded_rollout(inputs: BoundedFinalization) -> BoundedRolloutResult:
    """Append exact terminal/cleanup records and persist their typed receipt."""
    records = inputs.records
    state = "FAULT" if inputs.fault else "COMPLETE"
    writes = count_writes(inputs.bus)
    previous = inputs.previous
    if inputs.fault is not None and (not records or records[-1].get("kind") != "cycle_fault"):
        previous = append_record(
            records,
            {
                "kind": "cycle_fault",
                "cycle": len(inputs.command_ids),
                "fault_code": inputs.fault,
                "fault_detail": inputs.fault_detail,
                "write_count": writes,
                "retry_count": 0,
                "compensation_count": 0,
            },
            previous_digest=previous,
        )
    previous = append_record(
        records,
        {
            "kind": "terminal",
            "state": state,
            "fault_code": inputs.fault,
            "fault_detail": inputs.fault_detail,
            "write_count": writes,
            "error_count": inputs.error_count,
            "max_error_count": inputs.authorization.max_error_count,
            "authorization_digest": inputs.authorization.digest,
        },
        previous_digest=previous,
    )
    previous = append_record(
        records,
        {
            "kind": "cleanup",
            "state": state,
            "fault_code": inputs.fault,
            "fault_detail": inputs.fault_detail,
            "writer_closed": True,
            "write_count": writes,
            "retry_count": 0,
            "compensation_count": 0,
            "error_count": inputs.error_count,
        },
        previous_digest=previous,
    )
    result = BoundedRolloutResult(
        state,
        writes,
        tuple(inputs.command_ids),
        inputs.authorization.max_commands,
        inputs.fault,
        writes > 0,
        previous,
        inputs.error_count,
        inputs.authorization.max_error_count,
    )
    inputs.output.mkdir(parents=True, exist_ok=True)
    (inputs.output / "ledger.jsonl").write_text(serialize_ledger(records), encoding="utf-8")
    (inputs.output / "receipt.json").write_text(
        json.dumps(result.to_document(), sort_keys=True, indent=2), encoding="utf-8"
    )
    return result
