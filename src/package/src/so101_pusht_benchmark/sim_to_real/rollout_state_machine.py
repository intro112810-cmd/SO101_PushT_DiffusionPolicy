"""Pure one-way state transitions with process-local consistency coordination."""

from __future__ import annotations

import re

from .rollout_authority import TransitionResult
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_snapshot import RolloutSnapshot, RolloutState, TERMINAL_STATES

__all__ = ("RolloutSnapshot", "RolloutState", "advance")
_COMMAND_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.ASCII)
_VALID = frozenset(
    {
        (RolloutState.RECEIVED, RolloutState.VALIDATING),
        (RolloutState.VALIDATING, RolloutState.REJECTED),
        (RolloutState.VALIDATING, RolloutState.ARMED),
        (RolloutState.ARMED, RolloutState.DISPATCHING),
        (RolloutState.DISPATCHING, RolloutState.ACK_WAIT),
        (RolloutState.DISPATCHING, RolloutState.FAULT),
        (RolloutState.ACK_WAIT, RolloutState.OBSERVING),
        (RolloutState.ACK_WAIT, RolloutState.FAULT),
        (RolloutState.OBSERVING, RolloutState.COMPLETE),
        (RolloutState.OBSERVING, RolloutState.FAULT),
    }
)


def _command_id(value: str) -> str:
    """Accept one canonical ASCII token without normalizing untrusted text."""
    if len(value) > 128 or _COMMAND_PATTERN.fullmatch(value) is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "invalid command_id")
    return value


def _require_accepted(result: TransitionResult, detail: str) -> None:
    if result is TransitionResult.ACCEPTED:
        return
    if result in {TransitionResult.DUPLICATE_COMMAND, TransitionResult.DUPLICATE_OUTCOME}:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, detail)
    raise RolloutViolation(RolloutCode.R_STALE_TRANSITION, detail)


def advance(
    snapshot: RolloutSnapshot,
    target: RolloutState,
    *,
    command_id: str | None = None,
    code: RolloutCode | None = None,
) -> RolloutSnapshot:
    """Atomically consume the authoritative source revision before transition."""
    selected_id = command_id if command_id is not None else snapshot.command_id
    if selected_id is not None:
        selected_id = _command_id(selected_id)
    if snapshot.state in TERMINAL_STATES:
        raise RolloutViolation(RolloutCode.R_TERMINAL_STATE, snapshot.state.value)
    if (snapshot.state, target) not in _VALID:
        raise RolloutViolation(
            RolloutCode.R_INVALID_TRANSITION,
            f"{snapshot.state.value}->{target.value}",
        )
    terminal_code = code if target in TERMINAL_STATES else None
    if target in {RolloutState.REJECTED, RolloutState.FAULT} and terminal_code is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "terminal code")

    coordinator = snapshot.coordinator
    expected = (snapshot.revision, snapshot.state.value)
    dispatched = snapshot.dispatched_command_ids
    dispatch_revision = snapshot.dispatch_source_revision
    if target is RolloutState.DISPATCHING:
        if coordinator is None:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "transition coordinator")
        if selected_id is None:
            raise RolloutViolation(RolloutCode.R_MISSING, "command_id")
        if dispatched:
            raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, selected_id)
        _require_accepted(
            coordinator.consume_dispatch(snapshot.rollout_id, expected, selected_id),
            selected_id,
        )
        dispatched = dispatched | {selected_id}
        dispatch_revision = snapshot.revision
    elif snapshot.state is RolloutState.DISPATCHING and target in {
        RolloutState.ACK_WAIT,
        RolloutState.FAULT,
    }:
        if coordinator is None or selected_id is None or dispatch_revision is None:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "dispatch outcome")
        _require_accepted(
            coordinator.consume_outcome(
                snapshot.rollout_id,
                expected,
                dispatch_revision,
                selected_id,
                target.value,
            ),
            selected_id,
        )
    elif coordinator is not None:
        _require_accepted(
            coordinator.consume_transition(snapshot.rollout_id, expected, target.value),
            snapshot.rollout_id,
        )

    return RolloutSnapshot(
        snapshot.rollout_id,
        target,
        selected_id,
        dispatched,
        terminal_code,
        snapshot.revision + 1,
        dispatch_revision,
        coordinator,
    )
