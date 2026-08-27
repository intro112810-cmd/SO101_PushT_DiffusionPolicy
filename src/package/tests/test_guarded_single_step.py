"""Todo 17 guarded single-step first-rollout arming gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.arming import (
    ArmingCheckInput,
    ArmingResult,
    check_arming,
)
from so101_pusht_benchmark.sim_to_real.ledger_chain import canonical_hash
from so101_pusht_benchmark.sim_to_real.replay_types import CAMERA_REGISTRATION_DIGEST
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.single_step_fixture import physical_proposal

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
PROFILE = BENCHMARK / "configs/hardware/so101_real_v1.yaml"
POLICY = FIXTURES / "collision_approved_policy.yaml"
SHADOW = FIXTURES / "shadow_campaign.jsonl"
AUTH = FIXTURES / "single_step_authorization.json"
SUPERSEDED_AUTH = FIXTURES / "superseded_camera_single_step_authorization.json"
OPERATIONAL = FIXTURES / "single_step_operational"
SCRIPT = BENCHMARK / "scripts/check_guarded_single_step.py"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)

PROPOSAL_HASH = "6d05a6b3dc27ce01deba5804ab176e13c2afc52d5b506c4df582ce77ec9ad3c0"
POLICY_DIGEST = "f3d19716d616b3f587c04dc9896bbd68b4f4b162059938068278c68c49d2c5a7"
COMMAND_ID = "single-step-001"
FORBIDDEN = ("SOFollower", "sync_write", "Goal_Position", "send_action")


def _inputs(
    auth_path: Path | None = AUTH,
    operational_path: Path | None = OPERATIONAL,
) -> ArmingCheckInput:
    return ArmingCheckInput(
        profile_path=PROFILE,
        policy_path=POLICY,
        shadow_ledger_path=SHADOW,
        authorization_path=auth_path,
        operational_evidence_path=operational_path,
        now=NOW,
    )


def _load_auth() -> dict[str, object]:
    parsed: object = json.loads(AUTH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return cast("dict[str, object]", parsed)


def _auth_digest(doc: Mapping[str, object]) -> str:
    content = {
        key: value for key, value in doc.items() if key not in {"digest", "binding_signature"}
    }
    return canonical_hash(content)


def _operational_receipt(
    tmp_path: Path,
    filename: str,
    mutation: dict[str, object],
) -> Path:
    directory = tmp_path / "operational"
    shutil.copytree(OPERATIONAL, directory)
    path = directory / filename
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    document = cast("dict[str, object]", value)
    document.update(mutation)
    document["digest"] = canonical_hash(
        {key: item for key, item in document.items() if key != "digest"}
    )
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return directory


def _write_auth(path: Path, mutate: Callable[[dict[str, object]], None]) -> Path:
    doc = _load_auth()
    mutate(doc)
    doc["digest"] = _auth_digest(doc)
    path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _run_cli(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(BENCHMARK / "src"),
    }
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile",
            str(PROFILE),
            "--policy",
            str(POLICY),
            "--shadow-ledger",
            str(SHADOW),
            "--operational-evidence",
            str(OPERATIONAL),
            "--output",
            str(output),
            *extra,
        ],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# --- Given / When / Then unit gates ----------------------------------------


def test_superseded_camera_identity_and_signature_reject() -> None:
    superseded = json.loads(SUPERSEDED_AUTH.read_text(encoding="utf-8"))
    assert superseded["proposal_hash"] == (
        "ec0a554689bd514929a54043946e30cddccc2ba3f0f12a12669e5916034b3aa0"
    )
    with pytest.raises(RolloutViolation) as caught:
        load_single_step_authorization(SUPERSEDED_AUTH, now=NOW)

    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_regenerated_exact_camera_authorization_chain_passes() -> None:
    scene_pose = json.loads((FIXTURES / "single_step_scene_pose.json").read_text(encoding="utf-8"))
    authorization = load_single_step_authorization(AUTH, now=NOW)
    proposal = physical_proposal()

    assert scene_pose["camera_registration_digest"] == CAMERA_REGISTRATION_DIGEST
    assert authorization.proposal_hash == proposal.proposal_hash == PROPOSAL_HASH
    assert check_arming(_inputs()).proposal_hash == proposal.proposal_hash


def test_happy_arming_from_fixtures() -> None:
    """Given valid fixtures, When arming is checked, Then it arms without writes."""
    result = check_arming(_inputs())

    assert result.armed is True
    assert result.motor_writes_performed is False
    assert result.proposal_hash == PROPOSAL_HASH
    assert result.policy_digest == POLICY_DIGEST
    assert result.command_id == COMMAND_ID


def test_missing_authorization_rejects() -> None:
    """Given no authorization path, When arming is checked, Then it fails closed."""
    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(auth_path=None))
    assert caught.value.code is RolloutCode.R_MISSING
    assert "authorization missing" in str(caught.value)


def test_expired_authorization_rejects(tmp_path: Path) -> None:
    """Given an expired authorization, When arming is checked, Then it is stale."""
    auth = _write_auth(
        tmp_path / "expired.json", lambda doc: doc.update(expires_at="2026-08-23T11:00:00Z")
    )

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(auth_path=auth))
    assert caught.value.code is RolloutCode.R_STALE


def test_competing_holder_rejects(tmp_path: Path) -> None:
    """Given a competing holder, When arming is checked, Then ownership conflicts."""
    operational = _operational_receipt(
        tmp_path, "device_ownership.json", {"competing_holder": True}
    )

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(operational_path=operational))
    assert caught.value.code is RolloutCode.R_OWNERSHIP_CONFLICT


def test_non_exclusive_owner_rejects(tmp_path: Path) -> None:
    """Given non-exclusive ownership, When arming is checked, Then it conflicts."""
    operational = _operational_receipt(
        tmp_path, "device_ownership.json", {"exclusive_owner": False}
    )

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(operational_path=operational))
    assert caught.value.code is RolloutCode.R_OWNERSHIP_CONFLICT


def test_deadman_inactive_rejects(tmp_path: Path) -> None:
    """Given an inactive deadman, When arming is checked, Then it rejects."""
    operational = _operational_receipt(tmp_path, "interlock.json", {"deadman_active": False})

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(operational_path=operational))
    assert caught.value.code is RolloutCode.R_DEADMAN_INACTIVE


def test_stop_not_clear_rejects(tmp_path: Path) -> None:
    """Given a blocked stop, When arming is checked, Then it rejects."""
    operational = _operational_receipt(tmp_path, "interlock.json", {"stop_clear": False})

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(operational_path=operational))
    assert caught.value.code is RolloutCode.R_DEADMAN_INACTIVE


def test_command_budget_exhausted_rejects(tmp_path: Path) -> None:
    """Given a non-single-step budget, When arming is checked, Then it rejects."""
    auth = _write_auth(tmp_path / "budget.json", lambda doc: doc.update(command_budget=2))

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(auth_path=auth))
    assert caught.value.code is RolloutCode.R_BUDGET_EXHAUSTED


def test_modified_torque_state_rejects(tmp_path: Path) -> None:
    """Given a modified torque state, When arming is checked, Then it rejects."""
    operational = _operational_receipt(tmp_path, "torque_state.json", {"state": "modified"})

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(operational_path=operational))
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_mutated_authorization_digest_rejects(tmp_path: Path) -> None:
    """Given a drifted digest, When arming is checked, Then the hash mismatches."""
    doc = _load_auth()
    mutated = doc["digest"]
    assert isinstance(mutated, str)
    doc["digest"] = ("0" if mutated[0] != "0" else "1") + mutated[1:]
    auth = tmp_path / "mutated.json"
    auth.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(auth_path=auth))
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_proposal_hash_not_in_ledger_rejects(tmp_path: Path) -> None:
    """Given an unbound proposal hash, When arming is checked, Then it mismatches."""
    auth = _write_auth(tmp_path / "unbound.json", lambda doc: doc.update(proposal_hash="a" * 64))

    with pytest.raises(RolloutViolation) as caught:
        check_arming(_inputs(auth_path=auth))
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_receipt_never_reports_motor_writes() -> None:
    """Given a happy arming, When it arms, Then zero motor writes are reported."""
    result: ArmingResult = check_arming(_inputs())

    assert result.armed is True
    assert result.motor_writes_performed is False


# --- CLI surface ------------------------------------------------------------


def test_happy_cli_prints_armed_without_writes(tmp_path: Path) -> None:
    """Given the full happy command, When run, Then exit 0 with an armed receipt."""
    output = tmp_path / "arming.json"
    result = _run_cli(output, "--authorization", str(AUTH))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["armed"] is True
    assert payload["motor_writes_performed"] is False
    assert payload["evidence_scope"] == "test_fixture_only"
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_missing_authorization_cli_exits_2(tmp_path: Path) -> None:
    """Given no authorization, When the CLI runs, Then exit 2 with a clear reason."""
    output = tmp_path / "arming.json"
    result = _run_cli(output)

    assert result.returncode == 2
    assert "authorization missing" in result.stderr
    assert result.stdout == ""
    assert not output.exists()


# --- AST safety surface -----------------------------------------------------


def test_arming_surface_has_no_writer_symbols() -> None:
    """Given the module and CLI, When AST-scanned, Then no writer symbols exist."""
    paths = (
        BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/arming.py",
        SCRIPT,
    )
    missing = [str(path.relative_to(BENCHMARK)) for path in paths if not path.is_file()]
    assert not missing, f"missing arming surface: {missing}"

    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert all(symbol not in source for symbol in FORBIDDEN)
    assert "DirectBusWriter" not in source
