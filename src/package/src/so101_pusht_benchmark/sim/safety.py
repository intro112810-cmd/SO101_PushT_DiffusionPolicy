"""Latched simulation safety state machine and contact allowlist."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Fault(str, Enum):
    INVALID_ACTION = "invalid_action"
    INVALID_IK = "invalid_ik"
    FORBIDDEN_CONTACT = "forbidden_contact"
    NONFINITE_PHYSICS = "nonfinite_physics"
    RESET_EXHAUSTED = "reset_exhausted"
    TERMINAL = "terminal"
    COLLECTION_ABORT = "collection_abort"


@dataclass(slots=True)
class SafetyState:
    fault: Fault | None = None

    def latch(self, fault: Fault) -> None:
        self.fault = fault

    @property
    def safe(self) -> bool:
        return self.fault is None

    def reset(self) -> None:
        self.fault = None


def allowed_contact(first: str, second: str) -> bool:
    return frozenset((first, second)) in (
        frozenset(("push_t_bar", "table")),
        frozenset(("push_t_stem", "table")),
        frozenset(("pusher", "push_t_bar")),
        frozenset(("pusher", "push_t_stem")),
    )
