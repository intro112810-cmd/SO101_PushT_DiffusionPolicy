"""Durable signed-authority and budget-accounting records."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .bounded_authorization import BoundedAuthorization
from .bounded_execution import cartesian_target, swept_path_length
from .bounded_pipeline import PlannedCycle
from .ledger_chain import GENESIS_DIGEST
from .shadow_ledger import append_record


def record_bounded_authority(
    records: list[dict[str, object]], authorization: BoundedAuthorization
) -> str:
    """Persist the complete signed authority before any cycle."""
    return append_record(
        records,
        {
            "kind": "bounded_authorization",
            "authorization_digest": authorization.digest,
            "signed_authorization": authorization.signed_document,
            "policy_digest": authorization.policy_digest,
            "max_commands": authorization.max_commands,
            "max_duration_seconds": authorization.max_duration_seconds,
            "max_path_length_m": authorization.max_path_length_m,
            "max_error_count": authorization.max_error_count,
        },
        previous_digest=GENESIS_DIGEST,
    )


@dataclass(slots=True)
class BudgetRecorder:
    """Stateful recorder whose context is refreshed before each planned cycle."""

    records: list[dict[str, object]]
    cycle: int = 0
    command_count: int = 0
    elapsed_seconds: float = 0.0
    error_count: int = 0
    path_length_m: float = 0.0
    previous_target: tuple[float, float, float] | None = None

    def pre_cycle(
        self,
        previous: str,
        cycle: int,
        command_count: int,
        elapsed_seconds: float,
        error_count: int,
    ) -> str:
        """Persist usage observed before planning and retain cycle context."""
        self.cycle = cycle
        self.command_count = command_count
        self.elapsed_seconds = elapsed_seconds
        self.error_count = error_count
        return self._record(previous, "pre_cycle", 0.0, 0.0)

    def planned_path(
        self, previous: str, plan: PlannedCycle
    ) -> tuple[str, tuple[float, float, float]]:
        """Compute and persist exact cumulative path for one proposal."""
        swept = swept_path_length(plan)
        target = cartesian_target(plan)
        transition = (
            0.0 if self.previous_target is None else math.dist(self.previous_target, target)
        )
        self.path_length_m += swept + transition
        self.command_count += 1
        return self._record(previous, "post_plan", swept, transition), target

    def accept_target(self, target: tuple[float, float, float]) -> None:
        """Advance transition accounting only after a verified cycle."""
        self.previous_target = target

    def _record(self, previous: str, phase: str, swept: float, transition: float) -> str:
        return append_record(
            self.records,
            {
                "kind": "budget_accounting",
                "cycle": self.cycle,
                "phase": phase,
                "command_count": self.command_count,
                "elapsed_seconds": self.elapsed_seconds,
                "cumulative_path_m": self.path_length_m,
                "error_count": self.error_count,
                "swept_path_increment_m": swept,
                "target_transition_increment_m": transition,
            },
            previous_digest=previous,
        )
