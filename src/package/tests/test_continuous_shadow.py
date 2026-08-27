"""Todo 15: planner-complete continuous shadow with writer unavailable.

The orchestrator runs fresh acquisition, history assembly, transform, physical
IK, supervisor minting and ledger append without importing or invoking any
motor-write symbol. Stale campaigns latch HOLD; a valid fixture completes one
cycle and emits ``SHADOW_COMPLETE``.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.ledger_chain import (
    GENESIS_DIGEST,
    canonical_hash,
    replay_digest,
    verify_ledger,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import scene_pose_content_digest
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.shadow_campaign import (
    ShadowCampaignInput,
    run_shadow_campaign,
)
from so101_pusht_benchmark.sim_to_real import shadow_ledger
from so101_pusht_benchmark.sim_to_real.shadow_replay import verify_shadow_decision_ledger
from so101_pusht_benchmark.sim_to_real.shadow_types import FixtureClock, ShadowCampaignResult

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
SCRIPTS = BENCHMARK / "scripts"
MODULE = BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/shadow_campaign.py"
SCRIPT = SCRIPTS / "run_continuous_sim_to_real_shadow.py"
POLICY = FIXTURES / "collision_approved_policy.yaml"
SHADOW_CAMPAIGN = FIXTURES / "shadow_campaign"
PLANNER_COMPLETE_CAMPAIGN = FIXTURES / "planner_complete_shadow_campaign"
STALE_SHADOW_CAMPAIGN = FIXTURES / "stale_shadow_campaign"
LINEAGE_AUTHORITY = "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)

FORBIDDEN_NAMES = (
    "SOFollower",
    "sync_write",
    "Goal_Position",
    "send_action",
    "LerobotWriter",
    "motor_writer",
    "enable_torque",
    "Torque_Enable",
)
FORBIDDEN_MODULES = (
    "lerobot.motor_writer",
    "lerobot.teleoperators",
    "so101_pusht_benchmark.hardware_live",
)


def _ast_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _rehash_outer_chain(documents: list[dict[str, object]]) -> None:
    previous = GENESIS_DIGEST
    for sequence, document in enumerate(documents):
        document["sequence"] = sequence
        document["prev_digest"] = previous
        content = {key: value for key, value in document.items() if key != "digest"}
        previous = canonical_hash(content)
        document["digest"] = previous


def _run_cli(
    fixture: Path,
    output_dir: Path,
    *,
    policy_seed: int = 8,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PUSHT_SINGLE_CAM": "1",
        "PUSHT_LOCAL_BUDGET": "1",
        "PYTHONPATH": str(BENCHMARK / "src"),
    }
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            str(fixture),
            "--policy",
            str(POLICY),
            "--output-dir",
            str(output_dir),
            "--policy-seed",
            str(policy_seed),
        ],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _campaign_input(
    tmp_path: Path,
    *,
    fixture: Path = SHADOW_CAMPAIGN,
    policy_seed: int = 8,
) -> ShadowCampaignInput:
    return ShadowCampaignInput(
        fixture_dir=fixture,
        policy=load_fixture_safety_policy(POLICY, now=NOW),
        lineage_document=_json(FIXTURES / "lineage.json"),
        lineage_authority_digest=LINEAGE_AUTHORITY,
        joint_document=_json(FIXTURES / "joint-equivalence.json"),
        camera_document=_json(FIXTURES / "camera-registration.json"),
        camera_corpus=_json(FIXTURES / "camera_registration_valid" / "corpus.json"),
        source_frame_path=FIXTURES / "physical_frame.png",
        output_dir=tmp_path / "campaign",
        clock=FixtureClock(start=1_000_000.0, step=0.01),
        cycle_limit=1,
        policy_seed=policy_seed,
    )


# --- AST / actuation isolation -----------------------------------------------


def test_shadow_module_ast_has_zero_writer_symbols() -> None:
    names = _ast_names(MODULE)

    assert not (names & set(FORBIDDEN_NAMES))
    assert not (names & set(FORBIDDEN_MODULES))


def test_shadow_cli_ast_has_zero_writer_symbols() -> None:
    names = _ast_names(SCRIPT)

    assert not (names & set(FORBIDDEN_NAMES))
    assert not (names & set(FORBIDDEN_MODULES))


def test_production_shadow_route_fails_closed_without_bound_evidence(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--production-evidence-dir",
            str(tmp_path / "missing"),
            "--policy",
            str(POLICY),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 2
    assert "R_MISSING: production frozen-policy evidence directory is unavailable" in result.stderr
    assert not (tmp_path / "output").exists()


def test_production_campaign_persistence_preserves_receipt_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "production-shadow"
    output_dir.mkdir()
    observed_scopes: list[bool] = []

    def accept_scope(
        identity: shadow_ledger.ReceiptPathIdentity, *, production: bool
    ) -> shadow_ledger.ReceiptPathIdentity:
        observed_scopes.append(production)
        return identity

    monkeypatch.setattr(shadow_ledger, "validate_receipt_identity", accept_scope)
    records: list[dict[str, object]] = []
    ledger_digest = shadow_ledger.append_record(
        records,
        {"kind": "campaign_complete", "terminal_state": "SHADOW_COMPLETE"},
        previous_digest=GENESIS_DIGEST,
    )
    result = ShadowCampaignResult(
        terminal_state="SHADOW_COMPLETE",
        terminal_code="SHADOW_COMPLETE",
        cycles_completed=1,
        cycle_limit=1,
        ledger_digest=ledger_digest,
        motor_writes_performed=False,
        actuation_performed=False,
        writer_symbols=0,
        evidence_scope="production",
        policy_evidence="authentic_frozen_production",
        receipt_path=output_dir / "SHADOW_COMPLETE",
        ledger_path=output_dir / "ledger.jsonl",
    )

    shadow_ledger.persist_campaign(records, result, output_dir)

    assert observed_scopes == [True, True]


# --- GREEN: fixture campaign completes ---------------------------------------


def test_shadow_campaign_emits_shadow_complete(tmp_path: Path) -> None:
    output_dir = tmp_path / "shadow"
    result = _run_cli(PLANNER_COMPLETE_CAMPAIGN, output_dir, policy_seed=479235)

    assert result.returncode == 0, result.stderr
    assert "SHADOW_COMPLETE" in result.stdout
    receipt = _json(output_dir / "SHADOW_COMPLETE")
    assert receipt["terminal_state"] == "SHADOW_COMPLETE"
    assert receipt["cycles_completed"] == 1
    assert receipt["cycle_limit"] == 1
    assert receipt["motor_writes_performed"] is False
    assert receipt["actuation_performed"] is False
    assert receipt["writer_symbols"] == 0
    assert receipt["evidence_scope"] == "test_fixture_only"
    assert receipt["policy_evidence"] == "fixture_adapter_not_frozen_production"
    assert isinstance(receipt["ledger_digest"], str)
    assert len(receipt["ledger_digest"]) == 64


def test_current_fixture_campaign_characterization() -> None:
    document = _json(SHADOW_CAMPAIGN / "samples.json")
    samples = cast("list[dict[str, object]]", document["samples"])

    assert len(samples) == 2
    assert samples[0]["record_id"] == "sample-000"
    assert samples[1]["record_id"] == "sample-001"
    assert samples[1]["body_degrees"] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_current_fixture_no_longer_receives_invented_green_ik(tmp_path: Path) -> None:
    result = _run_cli(SHADOW_CAMPAIGN, tmp_path / "current")

    assert result.returncode == 3
    terminal = _json(tmp_path / "current" / "terminal_receipt.json")
    assert terminal["terminal_state"] == "HOLD"
    assert terminal["terminal_code"] == RolloutCode.R_CLIPPING_REQUIRED.value


def test_shadow_chain_binds_authenticated_scene_pose_to_exact_sample(tmp_path: Path) -> None:
    result = run_shadow_campaign(
        _campaign_input(
            tmp_path,
            fixture=PLANNER_COMPLETE_CAMPAIGN,
            policy_seed=479235,
        )
    )
    assert result.terminal_state == "SHADOW_COMPLETE"
    documents = [
        cast("dict[str, object]", json.loads(line))
        for line in result.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    sample = cast("list[dict[str, object]]", documents[0]["sample_records"])[-1]
    proposal = cast("dict[str, object]", documents[3]["ik_proposal"])
    pose = _json(PLANNER_COMPLETE_CAMPAIGN / "samples.json")["scene_pose"]
    assert isinstance(pose, dict)
    assert pose["sample_id"] == sample["record_id"]
    assert pose["sample_timestamp"] == sample["created_at"]
    assert proposal["scene_pose_digest"] == pose["digest"]
    assert proposal["collision_samples"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_id", "other-shadow-sample"), ("sample_timestamp", 1_000_000.011)],
)
def test_shadow_rehashed_scene_pose_identity_drift_holds(
    field: str, value: object, tmp_path: Path
) -> None:
    document = _json(PLANNER_COMPLETE_CAMPAIGN / "samples.json")
    pose = cast("dict[str, object]", document["scene_pose"])
    pose[field] = value
    pose["digest"] = scene_pose_content_digest(pose)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "samples.json").write_text(json.dumps(document), encoding="utf-8")

    result = run_shadow_campaign(
        _campaign_input(tmp_path / "run", fixture=fixture, policy_seed=479235)
    )

    assert result.terminal_state == "HOLD"
    assert result.terminal_code == RolloutCode.R_HASH_MISMATCH.value
    assert result.motor_writes_performed is False


def test_shadow_chain_has_no_injectable_fixed_green_planner() -> None:
    fields = ShadowCampaignInput.__dataclass_fields__
    cli_source = SCRIPT.read_text(encoding="utf-8")
    chain_source = MODULE.read_text(encoding="utf-8")

    assert "ik" not in fields
    assert "FrozenGreenIK" not in cli_source
    assert "build_physical_ik_planner" in chain_source


def test_shadow_campaign_ledger_verifies_and_replays_deterministically(
    tmp_path: Path,
) -> None:
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    first = _run_cli(PLANNER_COMPLETE_CAMPAIGN, output_a, policy_seed=479235)
    second = _run_cli(PLANNER_COMPLETE_CAMPAIGN, output_b, policy_seed=479235)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    ledger = output_a / "ledger.jsonl"
    assert ledger.exists()
    documents = [
        cast("dict[str, object]", json.loads(line)) for line in ledger.read_text().splitlines()
    ]
    verify_ledger(documents)
    verify_shadow_decision_ledger(documents)
    kinds = [document["kind"] for document in documents]
    assert kinds == [
        "samples",
        "inference",
        "cartesian_transform",
        "ik_proposal",
        "supervisor_decision",
        "campaign_complete",
        "cleanup",
    ]
    samples = documents[0]
    assert len(cast("list[dict[str, object]]", samples["sample_records"])) == 2
    inference = documents[1]
    inference_receipt = cast("dict[str, object]", inference["inference_receipt"])
    actions = cast("list[list[float]]", inference_receipt["action_chunk_float32_2d"])
    assert len(actions) == 8
    assert inference["selected_action_0"] == actions[0]
    assert inference["policy_evidence"] == "fixture_adapter_not_frozen_production"
    cartesian = cast("dict[str, object]", documents[2]["cartesian_receipt"])
    assert cartesian["transform_hash"] == documents[2]["transform_hash"]
    ik = cast("dict[str, object]", documents[3]["ik_proposal"])
    assert ik["proposal_hash"] == documents[3]["proposal_hash"]
    decision = documents[4]
    token = cast("dict[str, object]", decision["authorization_token"])
    assert token["proposal_hash"] == ik["proposal_hash"]
    digest_a = replay_digest(documents)
    documents_b = [
        cast("dict[str, object]", json.loads(line))
        for line in (output_b / "ledger.jsonl").read_text().splitlines()
    ]
    assert replay_digest(documents_b) == digest_a
    verify_shadow_decision_ledger(documents_b)

    tampered_action = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    action_receipt = cast("dict[str, object]", tampered_action[1]["inference_receipt"])
    action_chunk = cast("list[list[float]]", action_receipt["action_chunk_float32_2d"])
    action_chunk[0][0] += 0.01
    _rehash_outer_chain(tampered_action)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_action)

    tampered_sample = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    sample_records = cast("list[dict[str, object]]", tampered_sample[0]["sample_records"])
    sample_records[0]["digest"] = "f" * 64
    _rehash_outer_chain(tampered_sample)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_sample)

    tampered_transform = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    transform_receipt = cast("dict[str, object]", tampered_transform[2]["cartesian_receipt"])
    transform_receipt["transform_hash"] = "f" * 64
    _rehash_outer_chain(tampered_transform)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_transform)

    tampered_applied = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    applied_receipt = cast("dict[str, object]", tampered_applied[2]["cartesian_receipt"])
    applied_xyz = cast("list[float]", applied_receipt["applied_xyz"])
    applied_xyz[0] += 0.001
    _rehash_outer_chain(tampered_applied)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_applied)

    tampered_ik = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    ik_receipt = cast("dict[str, object]", tampered_ik[3]["ik_proposal"])
    body_degrees = cast("list[float]", ik_receipt["body_degrees"])
    body_degrees[0] += 1.0
    _rehash_outer_chain(tampered_ik)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_ik)

    tampered_cleanup = cast("list[dict[str, object]]", json.loads(json.dumps(documents)))
    tampered_cleanup[-1]["writer_closed"] = False
    _rehash_outer_chain(tampered_cleanup)
    with pytest.raises(RolloutViolation):
        verify_shadow_decision_ledger(tampered_cleanup)


# --- RED: stale campaign latches HOLD and writes nothing ----------------------


def test_stale_campaign_latches_hold_and_promotes_no_success(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "stale"
    result = _run_cli(STALE_SHADOW_CAMPAIGN, output_dir)

    assert result.returncode != 0
    assert not (output_dir / "SHADOW_COMPLETE").exists()
    terminal = _json(output_dir / "terminal_receipt.json")
    assert terminal["terminal_state"] == "HOLD"
    assert terminal["terminal_code"] == RolloutCode.R_STALE.value
    assert terminal["evidence_scope"] == "test_fixture_only"
    assert terminal["motor_writes_performed"] is False
    assert terminal["actuation_performed"] is False
    assert terminal["writer_symbols"] == 0

    ledger = output_dir / "ledger.jsonl"
    assert ledger.exists()
    ledger_text = ledger.read_text(encoding="utf-8")
    assert "sync_write" not in ledger_text
    assert "Goal_Position" not in ledger_text


def test_stale_campaign_module_latches_hold(tmp_path: Path) -> None:
    inputs = _campaign_input(tmp_path, fixture=STALE_SHADOW_CAMPAIGN)

    result = run_shadow_campaign(inputs)

    assert result.terminal_state == "HOLD"
    assert result.terminal_code == RolloutCode.R_STALE.value
    assert result.cycles_completed == 0
    assert result.motor_writes_performed is False
    assert result.actuation_performed is False
    assert not result.receipt_path.exists()


def test_fixture_clock_is_monotonic_and_deterministic() -> None:
    clock = FixtureClock(start=10.0, step=0.5)

    assert clock() == 10.0
    assert clock() == 10.5
    assert clock() == 11.0
    assert math.isfinite(clock())
