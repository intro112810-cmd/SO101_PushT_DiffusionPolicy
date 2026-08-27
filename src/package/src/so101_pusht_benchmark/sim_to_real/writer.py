"""The sole direct-bus write boundary for a guarded physical rollout."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from .authorization import AuthorizationToken
from .joint_mapping import JOINT_ORDER
from .physical_ik import PhysicalIKProposal

__all__ = ("DirectBusWriter", "DispatchIntent")
_GOAL_POSITION: Final = "Goal_Position"


class DirectBus(Protocol):
    """The only bus capability granted to the physical writer."""

    def connect(self) -> None:
        """Open the already-configured direct bus without changing its configuration."""

    def sync_write(self, register: str, payload: dict[str, float]) -> None:
        """Write one register payload to the configured body motors."""

    def disconnect(self, *, disable_torque: bool) -> None:
        """Release the bus while preserving the existing torque state."""


class DirectBusRobot(Protocol):
    """A robot wrapper exposing only its direct serial bus."""

    @property
    def bus(self) -> DirectBus:
        """Return the direct bus capability."""
        ...


class AuthorizationConsumer(Protocol):
    """The supervisor capability consumed at the dispatch boundary."""

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        """Consume one exact proposal-bound authorization token."""


@dataclass(frozen=True, slots=True)
class DispatchIntent:
    """The ledger content that must precede the one physical write attempt."""

    proposal_hash: str
    command_id: str
    body_degrees: tuple[float, float, float, float, float]


IntentAppender = Callable[[DispatchIntent], None]


class DirectBusWriter:
    """Dispatch exactly one authorized body-joint goal without retry or recovery motion."""

    def __init__(
        self,
        robot: DirectBusRobot,
        supervisor: AuthorizationConsumer,
        append_intent: IntentAppender,
    ) -> None:
        self._robot = robot
        self._supervisor = supervisor
        self._append_intent = append_intent

    def dispatch(
        self,
        token: AuthorizationToken,
        proposal: PhysicalIKProposal,
        command_id: str,
    ) -> None:
        """Consume, durably record, and attempt one direct body-only goal write."""
        self._supervisor.consume(token, proposal.proposal_hash, command_id)
        self._append_intent(
            DispatchIntent(
                proposal_hash=proposal.proposal_hash,
                command_id=command_id,
                body_degrees=proposal.body_degrees,
            )
        )
        payload: dict[str, float] = dict(zip(JOINT_ORDER, proposal.body_degrees, strict=True))
        self._robot.bus.connect()
        try:
            self._robot.bus.sync_write(_GOAL_POSITION, payload)
        finally:
            self._robot.bus.disconnect(disable_torque=False)
