"""Production bounded runner consumes fresh provider cycles one action at a time."""

from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from so101_pusht_benchmark.sim_to_real.authorization import AuthorizationClaim, mint_authorization
from so101_pusht_benchmark.sim_to_real.production_bounded import (
    ProductionBoundedBudget,
    ProductionCycle,
    execute_production_bounded,
)
from so101_pusht_benchmark.sim_to_real.production_single_step import ProductionSingleStepRuntime
from so101_pusht_benchmark.sim_to_real.single_step import FixtureBus
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.single_step_evidence import load_fixture_evidence_providers
from so101_pusht_benchmark.sim_to_real.single_step_fixture import fixture_evidence
from test_rollout_supervisor import FakeClock

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tests/fixtures/sim_to_real"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


class Robot:
    def __init__(self, bus: FixtureBus) -> None:
        self.bus = bus


class Provider:
    def __init__(self, cycle: ProductionCycle) -> None:
        self._item = cycle

    def next_cycle(self, index: int, previous_evidence_digest: str) -> ProductionCycle | None:
        assert len(previous_evidence_digest) == 64
        if index == 0:
            return self._item
        return None


def test_bounded_runner_stops_at_provider_end_without_reusing_action() -> None:
    ack, post, proposal = load_fixture_evidence_providers(F / "single_step_complete")
    auth = replace(
        load_single_step_authorization(F / "single_step_authorization.json", now=NOW),
        artifact_scope="production",
    )
    ev = fixture_evidence()
    token = mint_authorization(
        AuthorizationClaim(proposal.proposal_hash, auth.policy_digest, auth.command_id, 1001.0)
    )
    bus = FixtureBus()
    cycle = ProductionCycle(
        token,
        proposal,
        frozenset(s.digest for s in ev.samples),
        999.97,
        ProductionSingleStepRuntime(Robot(bus), ack, post),
    )
    result = execute_production_bounded(
        Provider(cycle), ProductionBoundedBudget(1, 10.0, 1.0, 0), clock=FakeClock()
    )
    assert result.state == "COMPLETE", result
    assert result.write_count == 1
    assert result.command_ids == (token.command_id,)


def test_missing_second_fresh_cycle_faults_instead_of_claiming_complete() -> None:
    class EmptyAfterFirst:
        def next_cycle(self, index: int, previous_evidence_digest: str) -> ProductionCycle | None:
            del index, previous_evidence_digest
            return None

    result = execute_production_bounded(
        EmptyAfterFirst(),
        ProductionBoundedBudget(2, 10.0, 1.0, 0),
        clock=FakeClock(),
    )
    assert result.state == "FAULT"
    assert result.fault_code == "R_MISSING"
    assert result.write_count == 0
