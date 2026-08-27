"""Todo 16 single-owner direct-bus writer contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from so101_pusht_benchmark.hardware_profile import HardwareProfile, load_hardware_profile
from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_arming
from so101_pusht_benchmark.sim_to_real.authorization import AuthorizationToken
from so101_pusht_benchmark.sim_to_real.direct_bus_adapter import build_real_direct_bus_adapter
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.writer import DirectBusWriter, DispatchIntent
from test_rollout_supervisor import FakeClock, build_evidence

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/sim_to_real"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


class FakeBus:
    def __init__(self) -> None:
        self.log: list[tuple[str, object]] = []

    def connect(self) -> None:
        self.log.append(("connect", None))

    def sync_write(self, register: str, payload: dict[str, float]) -> None:
        self.log.append(("sync_write", (register, dict(payload))))

    def disconnect(self, *, disable_torque: bool) -> None:
        self.log.append(("disconnect", disable_torque))


class FakeRobot:
    def __init__(self, bus: FakeBus) -> None:
        self.bus = bus

    def connect(self) -> None:
        raise AssertionError("SOFollower.connect is forbidden")

    def calibrate(self) -> None:
        raise AssertionError("SOFollower.calibrate is forbidden")

    def configure(self) -> None:
        raise AssertionError("SOFollower.configure is forbidden")

    def send_action(self) -> None:
        raise AssertionError("SOFollower.send_action is forbidden")


@dataclass
class FakeLedger:
    """Mutable fixture sink that records the required pre-write intent."""

    log: list[tuple[str, object]]

    def append(self, intent: DispatchIntent) -> None:
        self.log.append(("intent", intent))


class ThrowAfterWriteBus(FakeBus):
    def sync_write(self, register: str, payload: dict[str, float]) -> None:
        super().sync_write(register, payload)
        raise RuntimeError("transport outcome unknown")


def _authorized_writer(bus: FakeBus) -> tuple[DirectBusWriter, AuthorizationToken]:
    from so101_pusht_benchmark.sim_to_real.supervisor import RolloutSupervisor

    supervisor = RolloutSupervisor(FakeClock())
    evidence = build_evidence()
    token = supervisor.mint(evidence)
    ledger = FakeLedger(bus.log)
    return DirectBusWriter(FakeRobot(bus), supervisor, ledger.append), token


def _expected_payload() -> dict[str, float]:
    proposal = build_evidence().ik_proposal
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    return dict(zip(joints, proposal.body_degrees, strict=True))


def test_one_token_writes_goal_position_once() -> None:
    # Given
    bus = FakeBus()
    writer, token = _authorized_writer(bus)
    proposal = build_evidence().ik_proposal
    payload = _expected_payload()

    # When
    writer.dispatch(token, proposal, "command-1")

    # Then
    assert bus.log == [
        ("intent", DispatchIntent(proposal.proposal_hash, "command-1", proposal.body_degrees)),
        ("connect", None),
        ("sync_write", ("Goal_Position", payload)),
        ("disconnect", False),
    ]
    with pytest.raises(RolloutViolation) as caught:
        writer.dispatch(token, proposal, "command-1")
    assert caught.value.code is RolloutCode.R_DUPLICATE_DISPATCH
    assert bus.log.count(("sync_write", ("Goal_Position", payload))) == 1


def test_rejected_authorization_has_zero_writes() -> None:
    # Given
    bus = FakeBus()
    writer, token = _authorized_writer(bus)
    proposal = build_evidence().ik_proposal

    # When / Then
    with pytest.raises(RolloutViolation) as caught:
        writer.dispatch(token, proposal, "another-command")
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH
    assert bus.log == []


def test_throw_after_possible_write_never_retries() -> None:
    # Given
    bus = ThrowAfterWriteBus()
    writer, token = _authorized_writer(bus)
    proposal = build_evidence().ik_proposal

    # When / Then
    with pytest.raises(RuntimeError, match="transport outcome unknown"):
        writer.dispatch(token, proposal, "command-1")
    assert bus.log == [
        ("intent", DispatchIntent(proposal.proposal_hash, "command-1", proposal.body_degrees)),
        ("connect", None),
        ("sync_write", ("Goal_Position", _expected_payload())),
        ("disconnect", False),
    ]


def test_real_adapter_factory_is_unreachable_from_fixture_authorization() -> None:
    profile = load_hardware_profile(ROOT / "configs/hardware/so101_real_v1.yaml")
    authorization = load_single_step_authorization(
        FIXTURES / "single_step_authorization.json", now=NOW
    )
    armed = check_arming(
        ArmingCheckInput(
            ROOT / "configs/hardware/so101_real_v1.yaml",
            FIXTURES / "collision_approved_policy.yaml",
            FIXTURES / "shadow_campaign.jsonl",
            FIXTURES / "single_step_authorization.json",
            FIXTURES / "single_step_operational",
            NOW,
        )
    )
    factory_calls = 0

    def factory(profile: HardwareProfile) -> FakeRobot:
        del profile
        nonlocal factory_calls
        factory_calls += 1
        return FakeRobot(FakeBus())

    with pytest.raises(RolloutViolation) as caught:
        build_real_direct_bus_adapter(profile, authorization, armed, factory)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
    assert factory_calls == 0


def test_real_adapter_constructs_only_after_all_exact_production_bindings() -> None:
    profile = load_hardware_profile(ROOT / "configs/hardware/so101_real_v1.yaml")
    fixture_authorization = load_single_step_authorization(
        FIXTURES / "single_step_authorization.json", now=NOW
    )
    armed = check_arming(
        ArmingCheckInput(
            ROOT / "configs/hardware/so101_real_v1.yaml",
            FIXTURES / "collision_approved_policy.yaml",
            FIXTURES / "shadow_campaign.jsonl",
            FIXTURES / "single_step_authorization.json",
            FIXTURES / "single_step_operational",
            NOW,
        )
    )
    authorization = replace(fixture_authorization, artifact_scope="production")
    profile = replace(profile, policy_digest=authorization.policy_digest)
    robot = FakeRobot(FakeBus())
    factory_calls = 0

    def factory(profile: HardwareProfile) -> FakeRobot:
        del profile
        nonlocal factory_calls
        factory_calls += 1
        return robot

    assert build_real_direct_bus_adapter(profile, authorization, armed, factory) is robot
    assert factory_calls == 1
    assert robot.bus.log == []


def test_forbidden_lifecycle_and_registers_reject() -> None:
    # Given
    bus = FakeBus()
    writer, token = _authorized_writer(bus)
    proposal = build_evidence().ik_proposal

    # When
    writer.dispatch(token, proposal, "command-1")

    # Then
    writes = [entry for entry in bus.log if entry[0] == "sync_write"]
    assert writes == [("sync_write", ("Goal_Position", _expected_payload()))]
    assert bus.log[-1] == ("disconnect", False)
