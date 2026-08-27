"""Immutable state records that never authorize physical writes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from .rollout_authority import TransitionCoordinator, TransitionResult
from .rollout_codes import RolloutCode, RolloutViolation


class RolloutState(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    REJECTED = "REJECTED"
    ARMED = "ARMED"
    DISPATCHING = "DISPATCHING"
    ACK_WAIT = "ACK_WAIT"
    OBSERVING = "OBSERVING"
    COMPLETE = "COMPLETE"
    FAULT = "FAULT"


SnapshotValue: TypeAlias = str | int | None
__all__ = (
    "TODO2_PHYSICAL_AUTHORITY_BOUNDARY",
    "RolloutSnapshot",
    "RolloutState",
    "SnapshotValue",
)
TODO2_PHYSICAL_AUTHORITY_BOUNDARY = (
    "TODO2_STATE_ONLY__TODO13_SUPERVISOR_TOKEN_REQUIRED__TODO16_CONSUMES"
)
TERMINAL_STATES = frozenset({RolloutState.REJECTED, RolloutState.COMPLETE, RolloutState.FAULT})
IN_FLIGHT_STATES = frozenset(
    {RolloutState.DISPATCHING, RolloutState.ACK_WAIT, RolloutState.OBSERVING}
)
_PRE_DISPATCH = frozenset({RolloutState.RECEIVED, RolloutState.VALIDATING, RolloutState.ARMED})


@dataclass(frozen=True, slots=True)
class RolloutSnapshot:
    """Immutable state only; Todo 13/16 own all future writer authority."""

    rollout_id: str
    state: RolloutState
    command_id: str | None
    dispatched_command_ids: frozenset[str]
    terminal_code: RolloutCode | None
    revision: int
    dispatch_source_revision: int | None
    coordinator: TransitionCoordinator | None = field(default=None, repr=False, compare=False)

    @classmethod
    def received(
        cls,
        rollout_id: str,
        *,
        coordinator: TransitionCoordinator | None = None,
    ) -> RolloutSnapshot:
        if not rollout_id:
            raise RolloutViolation(RolloutCode.R_MISSING, "rollout_id")
        if (
            coordinator is not None
            and coordinator.register_rollout(rollout_id, RolloutState.RECEIVED.value)
            is not TransitionResult.ACCEPTED
        ):
            raise RolloutViolation(RolloutCode.R_STALE_TRANSITION, rollout_id)
        return cls(
            rollout_id=rollout_id,
            state=RolloutState.RECEIVED,
            command_id=None,
            dispatched_command_ids=frozenset(),
            terminal_code=None,
            revision=0,
            dispatch_source_revision=None,
            coordinator=coordinator,
        )

    def to_dict(self) -> dict[str, SnapshotValue]:
        return {
            "rollout_id": self.rollout_id,
            "state": self.state.value,
            "command_id": self.command_id,
            "dispatched_command_ids": "\n".join(sorted(self.dispatched_command_ids)),
            "terminal_code": None if self.terminal_code is None else self.terminal_code.value,
            "revision": self.revision,
            "coordinator_name": None if self.coordinator is None else self.coordinator.name,
            "dispatch_source_revision": self.dispatch_source_revision,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, SnapshotValue]) -> RolloutSnapshot:
        state_value = raw.get("state")
        if isinstance(state_value, str) and state_value in {
            state.value for state in IN_FLIGHT_STATES
        }:
            raise RolloutViolation(RolloutCode.R_INVALID_TRANSITION, "in-flight resume forbidden")
        try:
            rollout_id = raw["rollout_id"]
            command_id = raw["command_id"]
            dispatched = raw["dispatched_command_ids"]
            terminal_code = raw["terminal_code"]
            revision = raw["revision"]
            coordinator_name = raw["coordinator_name"]
            dispatch_revision = raw["dispatch_source_revision"]
        except KeyError as error:
            raise RolloutViolation(RolloutCode.R_MISSING, str(error)) from error
        if not isinstance(rollout_id, str) or not rollout_id or not isinstance(state_value, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "snapshot identity/state")
        if command_id is not None and not isinstance(command_id, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "command_id")
        if not isinstance(dispatched, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "dispatched_command_ids")
        dispatched_ids: frozenset[str] = (
            frozenset(dispatched.splitlines()) if dispatched else frozenset[str]()
        )
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise RolloutViolation(RolloutCode.R_INVALID_TRANSITION, "invalid revision")
        if coordinator_name is not None and not isinstance(coordinator_name, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "coordinator_name")
        if dispatch_revision is not None and not isinstance(dispatch_revision, int):
            raise RolloutViolation(RolloutCode.R_INVALID_TRANSITION, "dispatch revision")
        if terminal_code is not None and not isinstance(terminal_code, str):
            raise RolloutViolation(RolloutCode.R_MISSING, "terminal_code")
        try:
            state = RolloutState(state_value)
            code = None if terminal_code is None else RolloutCode(terminal_code)
        except ValueError as error:
            raise RolloutViolation(RolloutCode.R_MISSING, str(error)) from error
        _validate_resume(state, (command_id, dispatched_ids), (code, dispatch_revision))
        return cls(
            rollout_id,
            state,
            command_id,
            dispatched_ids,
            code,
            revision,
            dispatch_revision,
            None,
        )


def _validate_resume(
    state: RolloutState,
    identity: tuple[str | None, frozenset[str]],
    status: tuple[RolloutCode | None, int | None],
) -> None:
    command_id, dispatched = identity
    code, dispatch_revision = status
    pre_dispatch_valid = (
        state in _PRE_DISPATCH
        and command_id is None
        and not dispatched
        and code is None
        and dispatch_revision is None
    )
    rejected_valid = (
        state is RolloutState.REJECTED
        and command_id is None
        and not dispatched
        and code is not None
    )
    completed_valid = (
        state is RolloutState.COMPLETE
        and command_id is not None
        and dispatched == frozenset({command_id})
        and code is None
        and dispatch_revision is not None
    )
    fault_valid = (
        state is RolloutState.FAULT
        and code is not None
        and (
            (command_id is None and not dispatched)
            or (
                command_id is not None
                and dispatched == frozenset({command_id})
                and dispatch_revision is not None
            )
        )
    )
    if not (pre_dispatch_valid or rejected_valid or completed_valid or fault_valid):
        raise RolloutViolation(RolloutCode.R_INVALID_TRANSITION, "unreachable snapshot")
