"""Typed bounded rollout inputs and terminal receipt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .shadow_types import CampaignClock
from .single_step import FixtureBus


@dataclass(frozen=True, slots=True)
class BoundedRolloutInput:
    fixture_dir: Path
    authorization_path: Path
    policy_path: Path
    single_step_receipt_path: Path
    output_dir: Path
    now: datetime
    clock: CampaignClock
    robot: FixtureBus | None = None


@dataclass(frozen=True, slots=True)
class BoundedRolloutResult:
    state: str
    write_count: int
    command_ids: tuple[str, ...]
    max_commands: int
    fault_code: str | None
    motor_writes_performed: bool
    ledger_digest: str
    error_count: int
    max_error_count: int

    def to_document(self) -> dict[str, object]:
        return {
            "schema": 2,
            "mode": "sim_to_real_bounded_rollout",
            "state": self.state,
            "write_count": self.write_count,
            "command_ids": list(self.command_ids),
            "evidence_scope": "test_fixture_only",
            "max_commands": self.max_commands,
            "fault_code": self.fault_code,
            "motor_writes_performed": self.motor_writes_performed,
            "policy_evidence": "fixture_fake_bus_not_production",
            "ledger_digest": self.ledger_digest,
            "error_count": self.error_count,
            "max_error_count": self.max_error_count,
        }
