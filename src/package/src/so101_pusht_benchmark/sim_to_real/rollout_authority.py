"""Process-local consistency coordination for rollout transitions.

Coordinators serialize pure state history and one-call budgets inside one
process. They are not security tokens, hardware authority, or proof of caller
provenance. Structural implementations may drive the state machine. Todo 13
alone mints proposal-bound expiring supervisor tokens; Todo 16 consumes those
tokens at the physical writer boundary.
"""

from __future__ import annotations

from enum import Enum
import re
from threading import Lock
from typing import Protocol

from .rollout_codes import RolloutCode, RolloutViolation

DispatchKey = tuple[str, int, str]
RolloutRevision = tuple[int, str]
__all__ = (
    "ProcessTransitionCoordinator",
    "RolloutRevision",
    "TransitionCoordinator",
    "TransitionResult",
    "request_transition_coordinator",
)
_COORDINATOR_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)


class TransitionResult(str, Enum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE_COMMAND = "DUPLICATE_COMMAND"
    DUPLICATE_OUTCOME = "DUPLICATE_OUTCOME"
    STALE_TRANSITION = "STALE_TRANSITION"


class TransitionCoordinator(Protocol):
    """Coordinate state consistency without conveying physical authority."""

    @property
    def name(self) -> str: ...

    def register_rollout(self, rollout_id: str, state: str) -> TransitionResult: ...

    def consume_transition(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        target_state: str,
    ) -> TransitionResult: ...

    def consume_dispatch(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        command_id: str,
    ) -> TransitionResult: ...

    def consume_outcome(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        dispatch_revision: int,
        command_id: str,
        target_state: str,
    ) -> TransitionResult: ...


class ProcessTransitionCoordinator:
    """Thread-safe mutable state for one caller-defined coordination scope."""

    __slots__ = ("__commands", "__current", "__lock", "__name", "__open", "__outcomes")

    def __init__(self, name: str) -> None:
        if len(name) > 63 or _COORDINATOR_PATTERN.fullmatch(name) is None:
            raise RolloutViolation(RolloutCode.R_MISSING, "invalid coordinator name")
        self.__name = name
        self.__commands: set[str] = set()
        self.__current: dict[str, RolloutRevision] = {}
        self.__open: set[DispatchKey] = set()
        self.__outcomes: set[DispatchKey] = set()
        self.__lock = Lock()

    @property
    def name(self) -> str:
        return self.__name

    def register_rollout(self, rollout_id: str, state: str) -> TransitionResult:
        initial = (0, state)
        with self.__lock:
            current = self.__current.get(rollout_id)
            if current is None:
                self.__current[rollout_id] = initial
                return TransitionResult.ACCEPTED
            return (
                TransitionResult.ACCEPTED
                if current == initial
                else TransitionResult.STALE_TRANSITION
            )

    def consume_transition(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        target_state: str,
    ) -> TransitionResult:
        with self.__lock:
            if self.__current.get(rollout_id) != expected:
                return TransitionResult.STALE_TRANSITION
            self.__current[rollout_id] = (expected[0] + 1, target_state)
            return TransitionResult.ACCEPTED

    def consume_dispatch(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        command_id: str,
    ) -> TransitionResult:
        dispatch = (rollout_id, expected[0], command_id)
        with self.__lock:
            if self.__current.get(rollout_id) != expected:
                return TransitionResult.STALE_TRANSITION
            if command_id in self.__commands:
                return TransitionResult.DUPLICATE_COMMAND
            self.__current[rollout_id] = (expected[0] + 1, "DISPATCHING")
            self.__commands.add(command_id)
            self.__open.add(dispatch)
            return TransitionResult.ACCEPTED

    def consume_outcome(
        self,
        rollout_id: str,
        expected: RolloutRevision,
        dispatch_revision: int,
        command_id: str,
        target_state: str,
    ) -> TransitionResult:
        dispatch = (rollout_id, dispatch_revision, command_id)
        with self.__lock:
            if dispatch in self.__outcomes or dispatch not in self.__open:
                return TransitionResult.DUPLICATE_OUTCOME
            if self.__current.get(rollout_id) != expected:
                return TransitionResult.STALE_TRANSITION
            self.__current[rollout_id] = (expected[0] + 1, target_state)
            self.__outcomes.add(dispatch)
            self.__open.remove(dispatch)
            return TransitionResult.ACCEPTED


_COORDINATORS: dict[str, ProcessTransitionCoordinator] = {}
_COORDINATOR_LOCK = Lock()


def request_transition_coordinator(name: str) -> ProcessTransitionCoordinator:
    """Return one shared process coordinator for a convenience label."""
    if len(name) > 63 or _COORDINATOR_PATTERN.fullmatch(name) is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "invalid coordinator name")
    with _COORDINATOR_LOCK:
        existing = _COORDINATORS.get(name)
        if existing is not None:
            return existing
        coordinator = ProcessTransitionCoordinator(name)
        _COORDINATORS[name] = coordinator
        return coordinator
