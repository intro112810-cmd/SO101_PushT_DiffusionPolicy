"""Production single-step core executes one exact prepared command."""

from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from so101_pusht_benchmark.sim_to_real.authorization import AuthorizationClaim, mint_authorization
from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_arming
from so101_pusht_benchmark.sim_to_real.production_evidence import (
    DirectBusEvidenceConfig,
    DirectBusEvidenceProvider,
)
from so101_pusht_benchmark.sim_to_real.production_single_step import (
    PreparedProductionSingleStep,
    ProductionSingleStepRuntime,
    execute_production_single_step,
)
from so101_pusht_benchmark.sim_to_real.single_step import FixtureBus
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.single_step_evidence import load_fixture_evidence_providers
from so101_pusht_benchmark.sim_to_real.single_step_fixture import fixture_evidence

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tests/fixtures/sim_to_real"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


class Robot:
    def __init__(self, bus: FixtureBus) -> None:
        self.bus = bus


def test_prepared_production_single_step_writes_once_and_builds_promotable_receipt() -> None:
    ack, post, proposal = load_fixture_evidence_providers(F / "single_step_complete")
    auth = replace(
        load_single_step_authorization(F / "single_step_authorization.json", now=NOW),
        artifact_scope="production",
    )
    armed = check_arming(
        ArmingCheckInput(
            ROOT / "configs/hardware/so101_real_v1.yaml",
            F / "collision_approved_policy.yaml",
            F / "shadow_campaign.jsonl",
            F / "single_step_authorization.json",
            F / "single_step_operational",
            NOW,
        )
    )
    evidence = fixture_evidence()
    token = mint_authorization(
        AuthorizationClaim(proposal.proposal_hash, auth.policy_digest, auth.command_id, 1001.0)
    )
    bus = FixtureBus()
    prepared = PreparedProductionSingleStep(
        auth, armed, token, proposal, frozenset(sample.digest for sample in evidence.samples)
    )
    result = execute_production_single_step(
        prepared, ProductionSingleStepRuntime(Robot(bus), ack, post)
    )
    assert result["state"] == "COMPLETE"
    assert result["write_count"] == 1
    assert result["proposal_hash"] == proposal.proposal_hash
    assert sum(item[0] == "sync_write" for item in bus.log) == 1


class ReadbackBus:
    def __init__(self) -> None:
        self.connected = False
        self.reads: list[str] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self, *, disable_torque: bool) -> None:
        assert disable_torque is False
        self.connected = False

    def sync_read(
        self, register: str, motors: str | list[str] | None = None, *, normalize: bool = True
    ) -> dict[str, float]:
        del motors
        assert self.connected
        assert normalize
        self.reads.append(register)
        return {
            "shoulder_pan": 1.0,
            "shoulder_lift": 2.0,
            "elbow_flex": 3.0,
            "wrist_flex": 4.0,
            "wrist_roll": 5.0,
            "gripper": 9.0,
        }


class ReadbackRobot:
    def __init__(self, bus: ReadbackBus) -> None:
        self.bus = bus


def test_direct_bus_provider_reads_goal_then_fresh_position_without_writes() -> None:
    bus = ReadbackBus()
    times = iter((1000.0, 1000.1))
    provider = DirectBusEvidenceProvider(
        DirectBusEvidenceConfig(
            ReadbackRobot(bus),
            "a" * 64,
            "command-real",
            "b" * 64,
            (1.0, 2.0, 3.0, 4.0, 5.0),
            lambda: next(times),
            lambda: (1000.2, "c" * 64),
        )
    )
    ack = provider.acknowledge()
    post = provider.read_post_state()
    assert bus.reads == ["Goal_Position", "Present_Position"]
    assert ack.accepted_body_degrees == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert post.created_at == 1000.2
    assert post.frame_digest == "c" * 64
