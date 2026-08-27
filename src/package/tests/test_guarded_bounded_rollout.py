"""Todo 19 guarded bounded one-action receding-horizon rollout contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from so101_pusht_benchmark.sim_to_real.bounded_authorization import (
    load_bounded_authorization,
    verify_single_step_receipt,
)
from so101_pusht_benchmark.sim_to_real.authorization import AuthorizationClaim, mint_authorization
from so101_pusht_benchmark.sim_to_real.bounded_replay import verify_bounded_ledger
from so101_pusht_benchmark.sim_to_real.bounded_rollout import (
    BoundedRolloutInput,
    run_fixture_bounded_rollout,
)
from so101_pusht_benchmark.sim_to_real.ledger_chain import canonical_hash
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import scene_pose_content_digest
from so101_pusht_benchmark.sim_to_real.replay_policy import run_fixture_policy
from so101_pusht_benchmark.sim_to_real.replay_types import HistoryStep
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step import FixtureBus
from so101_pusht_benchmark.sim_to_real.shadow_types import FixtureClock

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
POLICY = FIXTURES / "collision_approved_policy.yaml"
AUTH = FIXTURES / "bounded_rollout_authorization.json"
SINGLE_STEP_RECEIPT = FIXTURES / "bounded_verified_single_step_receipt.json"
COMPLETE_FIXTURE = FIXTURES / "bounded_complete"
FAULT_FIXTURE = FIXTURES / "bounded_fault_after_two"
STATIC_FIXTURE = FIXTURES / "bounded_static_proposal"
STALE_FIXTURE = FIXTURES / "bounded_stale"
PROVIDER_FIXTURE = FIXTURES / "bounded_provider_modified"
TRACKING_FIXTURE = FIXTURES / "bounded_tracking_fault"
ONE_ERROR_FIXTURE = FIXTURES / "bounded_one_error"
ERROR_BREACH_FIXTURE = FIXTURES / "bounded_error_breach"
PATH_AUTH = FIXTURES / "bounded_path_budget_authorization.json"
TIME_AUTH = FIXTURES / "bounded_time_budget_authorization.json"
SCRIPT = BENCHMARK / "scripts/run_guarded_bounded_rollout.py"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
FORBIDDEN = ("SOFollower", "calibrate", "configure", "send_action")
BODY_JOINTS = {
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
}


def _inputs(
    fixture_dir: Path,
    authorization_path: Path,
    tmp_path: Path,
    *,
    robot: FixtureBus | None = None,
) -> BoundedRolloutInput:
    return BoundedRolloutInput(
        fixture_dir=fixture_dir,
        authorization_path=authorization_path,
        policy_path=POLICY,
        single_step_receipt_path=SINGLE_STEP_RECEIPT,
        output_dir=tmp_path,
        now=NOW,
        clock=FixtureClock(start=1000.0, step=0.01),
        robot=robot,
    )


def _writes(bus: FixtureBus) -> list[tuple[str, dict[str, float]]]:
    result: list[tuple[str, dict[str, float]]] = []
    for entry in bus.log:
        if entry[0] != "sync_write":
            continue
        value = entry[1]
        assert isinstance(value, tuple)
        result.append((value[0], value[1]))
    return result


def _run_cli(
    fixture_dir: Path, authorization_path: Path, output_dir: Path
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(BENCHMARK / "src"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            str(fixture_dir),
            "--authorization",
            str(authorization_path),
            "--single-step-receipt",
            str(SINGLE_STEP_RECEIPT),
            "--output-dir",
            str(output_dir),
        ],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_happy_three_cycles_never_exceeds_budget(tmp_path: Path) -> None:
    # Given: a complete fixture and a three-command bounded authorization.
    bus = FixtureBus()
    inputs = _inputs(COMPLETE_FIXTURE, AUTH, tmp_path, robot=bus)

    # When: the bounded rollout runs.
    result = run_fixture_bounded_rollout(inputs)

    # Then: exactly three verified cycles write once each and never exceed budget.
    assert result.state == "COMPLETE"
    assert result.write_count == 3
    assert result.command_ids == ("command-1", "command-2", "command-3")
    assert result.max_commands == 3
    assert result.fault_code is None
    assert result.motor_writes_performed is True
    assert len(_writes(bus)) == 3
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == result.to_document()
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert verify_bounded_ledger(records) == result.ledger_digest
    inferences = [record for record in records if record["kind"] == "inference"]
    assert len(inferences) == 3
    assert all(
        len(record["inference_receipt"]["action_chunk_float32_2d"]) == 8 for record in inferences
    )
    assert len({record["inference_digest"] for record in inferences}) == 3
    action_chunks = {
        json.dumps(record["inference_receipt"]["action_chunk_float32_2d"]) for record in inferences
    }
    assert len(action_chunks) == 3, "static seed-only fixture inference is forbidden"
    proposals = [record for record in records if record["kind"] == "ik_proposal"]
    assert len({record["proposal_hash"] for record in proposals}) == 3
    assert sum(record["kind"] == "discarded_actions" for record in records) == 3


def test_fixture_policy_same_seed_depends_on_pixels_and_state() -> None:
    import hashlib

    def step(identity: str, pixel: int, state_value: float) -> HistoryStep:
        image = np.full((96, 96, 3), pixel, dtype=np.uint8)
        state = np.full((5,), state_value, dtype=np.float32)
        return HistoryStep(
            identity,
            hashlib.sha256(identity.encode()).hexdigest(),
            hashlib.sha256((identity + "-frame").encode()).hexdigest(),
            hashlib.sha256(image.tobytes()).hexdigest(),
            hashlib.sha256(state.tobytes()).hexdigest(),
            image,
            state,
        )

    baseline = (step("a", 1, 0.1), step("b", 2, 0.2))
    pixel_changed = (baseline[0], step("c", 3, 0.2))
    state_changed = (baseline[0], step("d", 2, 0.3))
    first = run_fixture_policy(baseline, seed=123).actions
    assert np.array_equal(first, run_fixture_policy(baseline, seed=123).actions)
    assert not np.array_equal(first, run_fixture_policy(pixel_changed, seed=123).actions)
    assert not np.array_equal(first, run_fixture_policy(state_changed, seed=123).actions)


def test_outer_rehashed_forged_token_command_rejects(tmp_path: Path) -> None:
    result = run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, AUTH, tmp_path))
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    supervisor = next(record for record in records if record["kind"] == "supervisor_decision")
    token = supervisor["authorization_token"]
    forged = mint_authorization(
        AuthorizationClaim(
            proposal_hash=token["proposal_hash"],
            policy_digest=token["policy_digest"],
            command_id="forged-command",
            valid_until=token["valid_until"],
        )
    )
    supervisor["authorization_token"] = {
        "token_id": forged.token_id,
        "proposal_hash": forged.proposal_hash,
        "policy_digest": forged.policy_digest,
        "command_id": forged.command_id,
        "valid_until": forged.valid_until,
        "digest": forged.digest,
    }
    _rehash(records)
    with pytest.raises(RolloutViolation) as caught:
        verify_bounded_ledger(records)
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH
    assert result.write_count == 3


@pytest.mark.parametrize(
    "field",
    ["collision_samples", "model_digest", "policy_digest", "scene_pose_digest", "obstacles"],
)
def test_rehashed_bounded_collision_proof_tamper_rejects(field: str, tmp_path: Path) -> None:
    run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, AUTH, tmp_path))
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    ik_record = next(record for record in records if record["kind"] == "ik_proposal")
    proposal = cast("dict[str, object]", ik_record["ik_proposal"])
    if field == "collision_samples":
        samples = cast("list[dict[str, object]]", proposal[field])
        samples[0]["minimum_clearance_m"] = 0.5
    elif field == "obstacles":
        transforms = cast("list[list[object]]", proposal["obstacle_transforms"])
        pose = cast("list[float]", transforms[0][1])
        pose[0] += 0.01
    else:
        proposal[field] = "0" * 64
    _rehash(records)

    with pytest.raises(RolloutViolation) as caught:
        verify_bounded_ledger(records)

    assert caught.value.code == RolloutCode.R_HASH_MISMATCH


def test_outer_rehashed_cycle_one_proposal_reuse_in_cycle_two_rejects(tmp_path: Path) -> None:
    run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, AUTH, tmp_path))
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    proposals = [record for record in records if record["kind"] == "ik_proposal"]
    reused = deepcopy(proposals[0]["ik_proposal"])
    proposals[1]["ik_proposal"] = reused
    proposals[1]["proposal_hash"] = reused["proposal_hash"]
    supervisor = next(
        record
        for record in records
        if record.get("kind") == "supervisor_decision" and record.get("cycle") == 1
    )
    old_token = cast("dict[str, object]", supervisor["authorization_token"])
    token = mint_authorization(
        AuthorizationClaim(
            proposal_hash=cast("str", reused["proposal_hash"]),
            policy_digest=cast("str", reused["policy_digest"]),
            command_id="command-2",
            valid_until=cast("float", old_token["valid_until"]),
        )
    )
    supervisor["authorization_token"] = {
        "token_id": token.token_id,
        "proposal_hash": token.proposal_hash,
        "policy_digest": token.policy_digest,
        "command_id": token.command_id,
        "valid_until": token.valid_until,
        "digest": token.digest,
    }
    intent = next(
        record for record in records if record.get("kind") == "intent" and record.get("cycle") == 1
    )
    intent.update(
        proposal_hash=token.proposal_hash,
        token_digest=token.digest,
        token_id=token.token_id,
        policy_digest=token.policy_digest,
        valid_until=token.valid_until,
    )
    body = cast("list[float]", reused["body_degrees"])
    ack_record = next(
        record
        for record in records
        if record.get("kind") == "acknowledgement" and record.get("cycle") == 1
    )
    ack = cast("dict[str, object]", ack_record["evidence"])
    ack.update(proposal_hash=token.proposal_hash, accepted_body_degrees=body)
    ack["digest"] = canonical_hash({key: value for key, value in ack.items() if key != "digest"})
    post_record = next(
        record
        for record in records
        if record.get("kind") == "post_state" and record.get("cycle") == 1
    )
    post = cast("dict[str, object]", post_record["evidence"])
    post.update(acknowledgement_digest=ack["digest"], body_degrees=body)
    post["digest"] = canonical_hash({key: value for key, value in post.items() if key != "digest"})
    verified = next(
        record
        for record in records
        if record.get("kind") == "cycle_verified" and record.get("cycle") == 1
    )
    verified.update(proposal_hash=token.proposal_hash, post_state_digest=post["digest"])
    _rehash(records)

    with pytest.raises(RolloutViolation) as caught:
        verify_bounded_ledger(records)

    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


@pytest.mark.parametrize(
    ("fixture", "authorization"),
    [
        (COMPLETE_FIXTURE, PATH_AUTH),
        (COMPLETE_FIXTURE, TIME_AUTH),
        (ERROR_BREACH_FIXTURE, AUTH),
    ],
    ids=("path", "time", "error"),
)
def test_outer_rehashed_real_budget_fault_relabel_complete_rejects(
    fixture: Path, authorization: Path, tmp_path: Path
) -> None:
    run_fixture_bounded_rollout(_inputs(fixture, authorization, tmp_path))
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    terminal = next(record for record in records if record["kind"] == "terminal")
    cleanup = next(record for record in records if record["kind"] == "cleanup")
    terminal.update(state="COMPLETE", fault_code=None, fault_detail=None)
    cleanup.update(state="COMPLETE", fault_code=None, fault_detail=None)
    _rehash(records)

    with pytest.raises(RolloutViolation) as caught:
        verify_bounded_ledger(records)

    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_outer_rehashed_command_budget_breach_relabel_complete_rejects(tmp_path: Path) -> None:
    run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, AUTH, tmp_path))
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    terminal_index = next(i for i, record in enumerate(records) if record["kind"] == "terminal")
    last_usage = next(
        record
        for record in reversed(records[:terminal_index])
        if record["kind"] == "budget_accounting"
    )
    records.insert(
        terminal_index,
        {
            "kind": "budget_accounting",
            "cycle": 3,
            "phase": "pre_cycle",
            "command_count": 4,
            "elapsed_seconds": last_usage["elapsed_seconds"],
            "cumulative_path_m": last_usage["cumulative_path_m"],
            "error_count": 0,
            "swept_path_increment_m": 0.0,
            "target_transition_increment_m": 0.0,
        },
    )
    _rehash(records)

    with pytest.raises(RolloutViolation) as caught:
        verify_bounded_ledger(records)

    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_each_cycle_dispatches_once_with_unique_command_id_and_body_only(
    tmp_path: Path,
) -> None:
    # Given: one complete fixture and a fresh auditable bus.
    bus = FixtureBus()
    inputs = _inputs(COMPLETE_FIXTURE, AUTH, tmp_path, robot=bus)

    # When: the rollout runs three cycles.
    run_fixture_bounded_rollout(inputs)

    # Then: three body-only Goal_Position writes, no gripper, no extra register.
    writes = _writes(bus)
    assert [register for register, _payload in writes] == ["Goal_Position"] * 3
    for _register, payload in writes:
        assert set(payload) == BODY_JOINTS
        assert "gripper" not in payload
    # The writer never emits compensating or reverse motion: exactly one write per cycle.
    assert [entry[0] for entry in bus.log].count("sync_write") == 3


def test_fault_after_two_terminates_without_retry(tmp_path: Path) -> None:
    # Given: the fault fixture whose bus throws on the third dispatch.
    result = run_fixture_bounded_rollout(_inputs(FAULT_FIXTURE, AUTH, tmp_path))

    # When/Then: exactly two writes land, terminal FAULT, and no fourth attempt.
    assert result.state == "FAULT"
    assert result.write_count == 2
    assert result.fault_code == "F_PROVIDER_ERROR"
    assert result.command_ids == ("command-1", "command-2")
    assert result.motor_writes_performed is True


def test_signed_budget_is_never_exceeded(tmp_path: Path) -> None:
    # Given: more fresh cycles than the signed three-command authorization allows.
    manifest = json.loads((COMPLETE_FIXTURE / "manifest.json").read_text())
    manifest["cycles"].append(manifest["cycles"][2])
    fixture = tmp_path / "over-budget"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    bus = FixtureBus()

    # When: the rollout runs against the signed command budget.
    result = run_fixture_bounded_rollout(_inputs(fixture, AUTH, tmp_path / "out", robot=bus))

    # Then: no fourth cycle is observed or dispatched.
    assert result.state == "COMPLETE"
    assert result.write_count == 3
    assert result.max_commands == 3
    assert len(_writes(bus)) == 3


def test_missing_authorization_rejects_with_zero_writes(tmp_path: Path) -> None:
    # Given: an authorization path that does not exist.
    bus = FixtureBus()
    inputs = _inputs(COMPLETE_FIXTURE, tmp_path / "missing.json", tmp_path, robot=bus)

    # When/Then: the run fails closed before any writer is constructed.
    with pytest.raises(RolloutViolation) as caught:
        run_fixture_bounded_rollout(inputs)
    assert caught.value.code is RolloutCode.R_MISSING
    assert _writes(bus) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_id", "other-cycle-sample"), ("sample_timestamp", 1000.011)],
)
def test_rehashed_scene_pose_sample_identity_drift_rejects_before_write(
    field: str, value: object, tmp_path: Path
) -> None:
    manifest = cast(
        "dict[str, object]",
        json.loads((COMPLETE_FIXTURE / "manifest.json").read_text(encoding="utf-8")),
    )
    cycles = cast("list[dict[str, object]]", manifest["cycles"])
    pose = cast("dict[str, object]", cycles[0]["scene_pose"])
    pose[field] = value
    pose["digest"] = scene_pose_content_digest(pose)
    fixture = tmp_path / "identity-drift"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bus = FixtureBus()

    result = run_fixture_bounded_rollout(_inputs(fixture, AUTH, tmp_path / "out", robot=bus))

    assert result.state == "FAULT"
    assert result.fault_code == RolloutCode.R_HASH_MISMATCH.value
    assert result.write_count == 0
    assert _writes(bus) == []


def test_mutated_authorization_digest_rejects_zero_writes(tmp_path: Path) -> None:
    # Given: a bounded authorization whose content digest has been tampered with.
    doc = cast("dict[str, object]", json.loads(AUTH.read_text(encoding="utf-8")))
    digest = doc["digest"]
    assert isinstance(digest, str)
    doc["digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    auth = tmp_path / "tampered-authorization.json"
    auth.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    bus = FixtureBus()

    # When/Then: the tampered authorization is rejected before dispatch.
    with pytest.raises(RolloutViolation) as caught:
        run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, auth, tmp_path, robot=bus))
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH
    assert _writes(bus) == []


def test_authorization_parser_distinct_from_single_step() -> None:
    # Given: the bounded authorization document.
    authorization = load_bounded_authorization(
        AUTH,
        now=NOW,
        single_step_receipt_digest=verify_single_step_receipt(SINGLE_STEP_RECEIPT),
    )

    # Then: it is typed, three-command, and does not reuse single-step identity.
    assert authorization.max_commands == 3
    assert authorization.approved_by == "collision-fixture-owner@example.invalid"
    assert authorization.policy_digest == (
        "f3d19716d616b3f587c04dc9896bbd68b4f4b162059938068278c68c49d2c5a7"
    )


# --- CLI surface ------------------------------------------------------------


def test_happy_cli_exits_zero_with_three_verified_writes(tmp_path: Path) -> None:
    # Given: the full happy command.
    result = _run_cli(COMPLETE_FIXTURE, AUTH, tmp_path / "happy")

    # Then: exit 0 with three verified cycles and a durable receipt.
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state"] == "COMPLETE"
    assert payload["write_count"] == 3
    assert payload["command_ids"] == ["command-1", "command-2", "command-3"]


def test_fault_cli_exits_nonzero_with_two_writes_and_fault(tmp_path: Path) -> None:
    # Given: the fault fixture whose bus throws on the third dispatch.
    result = _run_cli(FAULT_FIXTURE, AUTH, tmp_path / "fault")

    # Then: nonzero terminal FAULT, exactly two writes, and no retry.
    assert result.returncode == 3, result.stderr
    payload = json.loads(result.stderr.splitlines()[-1])
    assert payload["state"] == "FAULT"
    assert payload["write_count"] == 2
    assert payload["fault_code"] == "F_PROVIDER_ERROR"
    assert payload["command_ids"] == ["command-1", "command-2"]


def test_missing_authorization_cli_exits_two_zero_writes(tmp_path: Path) -> None:
    # Given: a missing authorization path.
    output_dir = tmp_path / "missing"
    result = _run_cli(COMPLETE_FIXTURE, tmp_path / "does-not-exist.json", output_dir)

    # Then: exit 2, no receipt, and no output directory is created.
    assert result.returncode == 2
    assert "authorization missing" in result.stderr
    assert result.stdout == ""
    assert not output_dir.exists()


def test_static_reused_history_stops_before_second_write(tmp_path: Path) -> None:
    manifest = json.loads((COMPLETE_FIXTURE / "manifest.json").read_text())
    manifest["cycles"][1]["samples"] = manifest["cycles"][0]["samples"]
    fixture = tmp_path / "static"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    bus = FixtureBus()
    result = run_fixture_bounded_rollout(_inputs(fixture, AUTH, tmp_path / "out", robot=bus))
    assert result.state == "FAULT"
    assert result.write_count == 1
    assert result.fault_code == RolloutCode.R_DUPLICATE_SAMPLE.value
    assert len(_writes(bus)) == 1


def test_reused_bounded_authorization_token_stops_without_extra_write(tmp_path: Path) -> None:
    bus = FixtureBus()
    inputs = _inputs(COMPLETE_FIXTURE, AUTH, tmp_path / "first", robot=bus)
    run_fixture_bounded_rollout(inputs)
    with pytest.raises(RolloutViolation) as caught:
        run_fixture_bounded_rollout(_inputs(COMPLETE_FIXTURE, AUTH, tmp_path / "second", robot=bus))
    assert caught.value.code is RolloutCode.R_DUPLICATE_DISPATCH
    assert len(_writes(bus)) == 3


@pytest.mark.parametrize(
    ("fixture", "code", "writes"),
    [
        (STALE_FIXTURE, RolloutCode.R_STALE, 0),
        (STATIC_FIXTURE, RolloutCode.R_HASH_MISMATCH, 1),
        (PROVIDER_FIXTURE, RolloutCode.R_ACK_MISMATCH, 1),
        (TRACKING_FIXTURE, RolloutCode.R_POST_STATE_MISMATCH, 1),
    ],
)
def test_adversarial_cycle_faults_stop_exactly(
    fixture: Path, code: RolloutCode, writes: int, tmp_path: Path
) -> None:
    bus = FixtureBus()
    result = run_fixture_bounded_rollout(_inputs(fixture, AUTH, tmp_path, robot=bus))
    assert result.state == "FAULT"
    assert result.fault_code == code.value
    assert result.write_count == writes
    assert len(_writes(bus)) == writes
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert verify_bounded_ledger(records) == result.ledger_digest


@pytest.mark.parametrize(("authorization", "writes"), [(PATH_AUTH, 1), (TIME_AUTH, 0)])
def test_signed_path_and_time_budget_breaches_stop_exactly(
    authorization: Path, writes: int, tmp_path: Path
) -> None:
    bus = FixtureBus()
    result = run_fixture_bounded_rollout(
        _inputs(COMPLETE_FIXTURE, authorization, tmp_path, robot=bus)
    )
    assert result.state == "FAULT"
    assert result.fault_code == RolloutCode.R_BUDGET_EXHAUSTED.value
    assert result.write_count == writes
    assert len(_writes(bus)) == writes
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert verify_bounded_ledger(records) == result.ledger_digest


def test_real_mode_missing_authorization_blocks_before_adapter(tmp_path: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(BENCHMARK / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile",
            str(BENCHMARK / "configs/hardware/so101_real_v1.yaml"),
            "--authorization",
            str(tmp_path / "missing.json"),
            "--single-step-receipt",
            str(SINGLE_STEP_RECEIPT),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "authorization missing" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("fixture", "state", "errors", "writes"),
    [
        (COMPLETE_FIXTURE, "COMPLETE", 0, 3),
        (ONE_ERROR_FIXTURE, "COMPLETE", 1, 2),
        (ERROR_BREACH_FIXTURE, "FAULT", 2, 1),
    ],
)
def test_error_budget_zero_one_and_breach(
    fixture: Path, state: str, errors: int, writes: int, tmp_path: Path
) -> None:
    bus = FixtureBus()
    result = run_fixture_bounded_rollout(_inputs(fixture, AUTH, tmp_path, robot=bus))
    assert result.state == state
    assert result.error_count == errors
    assert result.write_count == writes
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert verify_bounded_ledger(records) == result.ledger_digest
    assert len(_writes(bus)) == writes


def test_unverified_single_step_receipt_blocks_before_writer(tmp_path: Path) -> None:
    receipt = tmp_path / "single.json"
    receipt.write_text('{"state":"COMPLETE","write_count":1}')
    bus = FixtureBus()
    inputs = _inputs(COMPLETE_FIXTURE, AUTH, tmp_path / "out", robot=bus)
    inputs = inputs.__class__(
        fixture_dir=inputs.fixture_dir,
        authorization_path=inputs.authorization_path,
        policy_path=inputs.policy_path,
        single_step_receipt_path=receipt,
        output_dir=inputs.output_dir,
        now=inputs.now,
        clock=inputs.clock,
        robot=bus,
    )
    with pytest.raises(RolloutViolation):
        run_fixture_bounded_rollout(inputs)
    assert _writes(bus) == []


def _rehash(records: list[dict[str, object]]) -> None:
    from so101_pusht_benchmark.sim_to_real.ledger_chain import GENESIS_DIGEST
    from so101_pusht_benchmark.sim_to_real.shadow_ledger import append_record

    contents = [
        {
            key: value
            for key, value in record.items()
            if key not in {"digest", "prev_digest", "sequence"}
        }
        for record in records
    ]
    records.clear()
    previous = GENESIS_DIGEST
    for content in contents:
        previous = append_record(records, content, previous_digest=previous)


# --- AST safety surface -----------------------------------------------------


def test_bounded_surface_has_no_forbidden_lifecycle_symbols() -> None:
    """Given the module and CLI, When AST-scanned, Then no lifecycle calls exist."""
    import ast

    paths = (
        BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/bounded_rollout.py",
        BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/bounded_authorization.py",
        SCRIPT,
    )
    missing = [str(path.relative_to(BENCHMARK)) for path in paths if not path.is_file()]
    assert not missing, f"missing bounded rollout surface: {missing}"

    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert all(symbol not in source for symbol in FORBIDDEN)

    identifiers: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            if isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
    assert "compensat" not in {"compensat" for name in identifiers if "compensat" in name}
    assert "reverse" not in {name for name in identifiers if "reverse" in name}
