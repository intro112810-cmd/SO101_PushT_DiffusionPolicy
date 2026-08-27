"""Direct-bus readback providers for production command evidence."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from .joint_mapping import JOINT_ORDER
from .ledger_chain import canonical_hash
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_evidence import AcknowledgementEvidence, PostStateEvidence


class ReadbackBus(Protocol):
    def connect(self) -> None: ...
    def sync_read(
        self, register: str, motors: str | list[str] | None = None, *, normalize: bool = True
    ) -> Mapping[str, int | float]: ...
    def disconnect(self, *, disable_torque: bool) -> None: ...


class ReadbackRobot(Protocol):
    @property
    def bus(self) -> ReadbackBus: ...


FrameEvidenceSource = Callable[[], tuple[float, str]]
Clock = Callable[[], float]


def _body(
    values: Mapping[str, int | float], label: str
) -> tuple[float, float, float, float, float]:
    expected = frozenset((*JOINT_ORDER, "gripper"))
    if frozenset(values) != expected:
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, f"{label} motor set")
    result = tuple(float(values[name]) for name in JOINT_ORDER)
    return cast("tuple[float,float,float,float,float]", result)


@dataclass(frozen=True, slots=True)
class DirectBusEvidenceConfig:
    robot: ReadbackRobot
    provider_digest: str
    command_id: str
    proposal_hash: str
    expected_body: tuple[float, float, float, float, float]
    clock: Clock
    frame_source: FrameEvidenceSource


class DirectBusEvidenceProvider:
    """Read exact setpoint acknowledgement then a newer measured state."""

    def __init__(self, config: DirectBusEvidenceConfig) -> None:
        self._robot = config.robot
        self._provider_digest = config.provider_digest
        self._command_id = config.command_id
        self._proposal_hash = config.proposal_hash
        self._expected_body = config.expected_body
        self._clock = config.clock
        self._frame_source = config.frame_source
        self._ack: AcknowledgementEvidence | None = None

    def _read(self, register: str) -> tuple[float, float, float, float, float]:
        self._robot.bus.connect()
        try:
            return _body(self._robot.bus.sync_read(register, normalize=True), register)
        finally:
            self._robot.bus.disconnect(disable_torque=False)

    def acknowledge(self) -> AcknowledgementEvidence:
        body = self._read("Goal_Position")
        observed = self._clock()
        content = {
            "accepted_body_degrees": list(body),
            "command_id": self._command_id,
            "observed_at": observed,
            "proposal_hash": self._proposal_hash,
            "provider_digest": self._provider_digest,
        }
        evidence = AcknowledgementEvidence(
            self._command_id,
            self._proposal_hash,
            observed,
            self._provider_digest,
            body,
            canonical_hash(content),
        )
        self._ack = evidence
        return evidence

    def read_post_state(self) -> PostStateEvidence:
        if self._ack is None:
            raise RolloutViolation(
                RolloutCode.R_POST_STATE_MISSING, "acknowledgement required first"
            )
        body = self._read("Present_Position")
        created_at, frame_digest = self._frame_source()
        sample_digest = canonical_hash(
            {"body_degrees": list(body), "created_at": created_at, "frame_digest": frame_digest}
        )
        content = {
            "acknowledgement_digest": self._ack.digest,
            "body_degrees": list(body),
            "command_id": self._command_id,
            "created_at": created_at,
            "frame_digest": frame_digest,
            "sample_digest": sample_digest,
        }
        return PostStateEvidence(
            self._command_id,
            self._ack.digest,
            created_at,
            sample_digest,
            frame_digest,
            body,
            canonical_hash(content),
        )
