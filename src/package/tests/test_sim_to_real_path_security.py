"""Adversarial filesystem tests for guarded rollout durable evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.bounded_execution import DurableIntentLedger
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.session_manifest import (
    CANONICAL_SESSION_MEMBERS,
    write_session_manifest,
)
from so101_pusht_benchmark.sim_to_real.shadow_ledger import LedgerDocument, write_shadow_receipt
from so101_pusht_benchmark.sim_to_real.shadow_types import (
    FixtureClock,
    ShadowCampaignInput,
    ShadowCampaignResult,
)
from so101_pusht_benchmark.sim_to_real.writer import DispatchIntent

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
SCRIPTS = BENCHMARK / "scripts"
OUTSIDE_BYTES = b"outside-sentinel\n"


def _intent() -> DispatchIntent:
    return DispatchIntent(
        proposal_hash="a" * 64,
        command_id="command-path-attack",
        body_degrees=(0.1, 0.2, 0.3, 0.4, 0.5),
    )


def _document(path: Path) -> LedgerDocument:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("LedgerDocument", raw)


def _campaign_input(output: Path) -> ShadowCampaignInput:
    return ShadowCampaignInput(
        fixture_dir=FIXTURES / "planner_complete_shadow_campaign",
        policy=load_fixture_safety_policy(
            FIXTURES / "collision_approved_policy.yaml",
            now=datetime(2026, 8, 23, 12, tzinfo=timezone.utc),
        ),
        lineage_document=_document(FIXTURES / "lineage.json"),
        lineage_authority_digest=(
            "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"
        ),
        joint_document=_document(FIXTURES / "joint-equivalence.json"),
        camera_document=_document(FIXTURES / "camera-registration.json"),
        camera_corpus=_document(FIXTURES / "camera_registration_valid/corpus.json"),
        source_frame_path=FIXTURES / "physical_frame.png",
        output_dir=output,
        clock=FixtureClock(start=1_000_000.0, step=0.01),
        cycle_limit=1,
        policy_seed=479235,
    )


def _shadow_result(output: Path) -> ShadowCampaignResult:
    return ShadowCampaignResult(
        terminal_state="SHADOW_COMPLETE",
        terminal_code="SHADOW_COMPLETE",
        cycles_completed=1,
        cycle_limit=1,
        ledger_digest="b" * 64,
        motor_writes_performed=False,
        actuation_performed=False,
        writer_symbols=0,
        evidence_scope="test_fixture_only",
        policy_evidence="fixture_adapter_not_frozen_production",
        receipt_path=output / "SHADOW_COMPLETE",
        ledger_path=output / "ledger.jsonl",
    )


def _run_replay(output: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PUSHT_SINGLE_CAM": "1",
        "PUSHT_LOCAL_BUDGET": "1",
        "PYTHONPATH": str(BENCHMARK / "src"),
    }
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_sim_to_real_replay.py"),
            "--samples",
            str(FIXTURES / "synchronized_samples.json"),
            "--lineage",
            str(FIXTURES / "lineage.json"),
            "--joint",
            str(FIXTURES / "joint-equivalence.json"),
            "--camera",
            str(FIXTURES / "camera-registration.json"),
            "--output",
            str(output),
        ],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _stage_manifest_session(root: Path) -> None:
    shutil.copytree(FIXTURES / "complete_session", root)
    (root / "session_manifest.json").unlink()
    for member in CANONICAL_SESSION_MEMBERS:
        target = root / member
        if target.exists():
            continue
        if member == "approved_policy.yaml":
            shutil.copy2(FIXTURES / "approved_policy.yaml", target)
        elif member == "single_step_authorization.json":
            shutil.copy2(FIXTURES / "single_step_authorization.json", target)
        elif member == "bounded_authorization.json":
            shutil.copy2(FIXTURES / "bounded_rollout_authorization.json", target)
        elif member.endswith(".jsonl"):
            target.write_text(json.dumps({"kind": member}) + "\n", encoding="utf-8")
        else:
            target.write_text(
                json.dumps({"evidence_scope": "test_fixture_only", "fixture_only": True}) + "\n",
                encoding="utf-8",
            )


def test_direct_intents_symlink_cannot_append_outside_root(tmp_path: Path) -> None:
    output = tmp_path / "bounded"
    output.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    (output / "intents.jsonl").symlink_to(outside)

    with pytest.raises((OSError, ValueError, RolloutViolation)):
        DurableIntentLedger(output / "intents.jsonl").append(_intent())

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert (output / "intents.jsonl").is_symlink()
    assert not (output / "receipt.json").exists()


def test_intent_leaf_replacement_fails_closed_without_appending_attacker_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded"
    output.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    ledger = DurableIntentLedger(output / "intents.jsonl")
    ledger.append(_intent())
    (output / "intents.jsonl").unlink()
    (output / "intents.jsonl").symlink_to(outside)

    with pytest.raises((OSError, ValueError, RolloutViolation)):
        ledger.append(_intent())

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert not (output / "receipt.json").exists()


def test_nonregular_intent_leaf_is_rejected_without_removal(tmp_path: Path) -> None:
    output = tmp_path / "bounded"
    output.mkdir()
    nonregular = output / "intents.jsonl"
    nonregular.mkdir()

    with pytest.raises((OSError, ValueError, RolloutViolation)):
        DurableIntentLedger(nonregular).append(_intent())

    assert nonregular.is_dir()
    assert not (output / "receipt.json").exists()


def test_direct_shadow_ledger_dangling_symlink_cannot_escape_root(tmp_path: Path) -> None:
    from so101_pusht_benchmark.sim_to_real import shadow_campaign

    inputs = _campaign_input(tmp_path / "campaign")
    output = inputs.output_dir
    output.mkdir()
    outside = tmp_path / "outside-ledger.jsonl"
    (output / "ledger.jsonl").symlink_to(outside)

    with pytest.raises((OSError, RolloutViolation)):
        shadow_campaign.run_shadow_campaign(inputs)

    assert not outside.exists()
    assert (output / "ledger.jsonl").is_symlink()
    assert not (output / "SHADOW_COMPLETE").exists()
    assert not (output / "terminal_receipt.json").exists()


def test_predictable_shadow_receipt_temp_symlink_cannot_redirect_bytes(tmp_path: Path) -> None:
    output = tmp_path / "shadow"
    output.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    (output / ".SHADOW_COMPLETE.tmp").symlink_to(outside)

    with pytest.raises((OSError, RolloutViolation)):
        write_shadow_receipt(_shadow_result(output), output)

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert not (output / "SHADOW_COMPLETE").exists()


def test_shadow_receipt_temp_attack_cleans_owned_ledger(tmp_path: Path) -> None:
    from so101_pusht_benchmark.sim_to_real.shadow_campaign import run_shadow_campaign

    inputs = _campaign_input(tmp_path / "campaign")
    inputs.output_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    (inputs.output_dir / ".SHADOW_COMPLETE.tmp").symlink_to(outside)

    with pytest.raises((OSError, RolloutViolation)):
        run_shadow_campaign(inputs)

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert not (inputs.output_dir / "ledger.jsonl").exists()
    assert not (inputs.output_dir / "SHADOW_COMPLETE").exists()


def test_destination_replacement_race_fails_without_unlinking_attacker_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from so101_pusht_benchmark.sim_to_real import secure_io

    output = tmp_path / "shadow"
    output.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    original_link = secure_io.os.link

    def insert_attacker_leaf(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        (output / destination).symlink_to(outside)
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(secure_io.os, "link", insert_attacker_leaf)

    with pytest.raises((OSError, RolloutViolation)):
        write_shadow_receipt(_shadow_result(output), output)

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert (output / "SHADOW_COMPLETE").is_symlink()
    assert not (output / ".SHADOW_COMPLETE.tmp").exists()


def test_predictable_replay_temp_symlink_cannot_redirect_bytes(tmp_path: Path) -> None:
    output = tmp_path / "inference.json"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    (tmp_path / ".inference.json.tmp").symlink_to(outside)

    result = _run_replay(output)

    assert result.returncode != 0
    assert outside.read_bytes() == OUTSIDE_BYTES
    assert not output.exists()


def test_manifest_partial_replacement_race_cannot_redirect_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from so101_pusht_benchmark.sim_to_real import session_manifest

    session = tmp_path / "session"
    _stage_manifest_session(session)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(OUTSIDE_BYTES)
    original = session_manifest.derive_session_evidence_scope

    def replace_partial(root: Path) -> str:
        scope = original(root)
        (root / ".session_manifest.json.partial").symlink_to(outside)
        return scope

    monkeypatch.setattr(session_manifest, "derive_session_evidence_scope", replace_partial)

    with pytest.raises((OSError, RolloutViolation)):
        write_session_manifest(session, session_id="path-race-fixture")

    assert outside.read_bytes() == OUTSIDE_BYTES
    assert not (session / "session_manifest.json").exists()
