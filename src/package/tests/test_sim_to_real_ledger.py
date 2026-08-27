"""Hash-chained rollout ledger and deterministic non-actuating replay."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.ledger_chain import (
    LedgerDigest,
    LedgerViolation,
    replay_digest,
    verify_ledger,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
COMPLETE = FIXTURES / "complete-ledger.jsonl"
TAMPERED = FIXTURES / "tampered-ledger.jsonl"
SCRIPT = BENCHMARK / "scripts/verify_sim_to_real_ledger.py"
GENESIS = "0" * 64

FORBIDDEN = ("SOFollower", "sync_write", "Goal_Position", "send_action", "np.clip", "sleep")


def _records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        records.append(cast("dict[str, object]", parsed))
    return records


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def _run_cli(ledger: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(BENCHMARK / "src"),
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--ledger", str(ledger), "--replay"],
        cwd=BENCHMARK,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture(scope="session")
def complete_records() -> list[dict[str, object]]:
    return _records(COMPLETE)


# --- RED: tamper detection --------------------------------------------------


def test_truncated_ledger_rejects(tmp_path: Path) -> None:
    lines = COMPLETE.read_text(encoding="utf-8").splitlines(keepends=True)
    _write_lines(tmp_path / "truncated.jsonl", lines[:-1])
    with pytest.raises(LedgerViolation) as caught:
        verify_ledger(_records(tmp_path / "truncated.jsonl"))
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_reordered_records_reject(complete_records: list[dict[str, object]]) -> None:
    first, second, rest = complete_records[0], complete_records[1], complete_records[2:]
    with pytest.raises(LedgerViolation) as caught:
        verify_ledger([second, first, *rest])
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_single_byte_mutation_rejects(tmp_path: Path) -> None:
    lines = COMPLETE.read_text(encoding="utf-8").splitlines()
    first = list(lines[1])
    index = first.index("r")
    first[index] = "z"
    lines[1] = "".join(first)
    _write_lines(tmp_path / "mutated.jsonl", [line + "\n" for line in lines])
    with pytest.raises(LedgerViolation) as caught:
        verify_ledger(_records(tmp_path / "mutated.jsonl"))
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_dispatch_before_intent_rejects(complete_records: list[dict[str, object]]) -> None:
    intent = next(record for record in complete_records if record["kind"] == "intent")
    without_intent = [record for record in complete_records if record is not intent]
    with pytest.raises(LedgerViolation) as caught:
        verify_ledger(without_intent)
    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_genesis_prev_digest_is_zeros(complete_records: list[dict[str, object]]) -> None:
    assert complete_records[0]["prev_digest"] == GENESIS


# --- GREEN: complete chain and replay ---------------------------------------


def test_complete_fixture_verifies_chain(complete_records: list[dict[str, object]]) -> None:
    digest = verify_ledger(complete_records)
    assert digest == complete_records[-1]["digest"]
    kinds = [record["kind"] for record in complete_records]
    assert kinds[-1] == "cleanup"
    assert "intent" in kinds
    assert "dispatch_status" in kinds
    assert kinds.index("intent") < kinds.index("dispatch_status")


def test_replay_digest_is_stable_across_two_runs() -> None:
    first = _run_cli(COMPLETE)
    second = _run_cli(COMPLETE)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    assert first_payload["valid"] is True
    assert first.stdout == second.stdout
    assert first_payload["replay_digest"] == second_payload["replay_digest"]
    assert len(first_payload["replay_digest"]) == 64


def test_tampered_fixture_cli_rejects() -> None:
    result = _run_cli(TAMPERED)
    assert result.returncode != 0
    assert "ledger hash mismatch" in result.stderr
    assert result.stdout == ""


def test_replay_module_has_no_hardware_dependency() -> None:
    module_path = BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/ledger_chain.py"
    script_path = SCRIPT
    source = "\n".join(path.read_text(encoding="utf-8") for path in (module_path, script_path))
    assert all(symbol not in source for symbol in FORBIDDEN)


def test_replay_digest_matches_concatenated_record_digests(
    complete_records: list[dict[str, object]],
) -> None:
    import hashlib

    combined = hashlib.sha256()
    for record in complete_records:
        combined.update(str(record["digest"]).encode("ascii"))
    digest = LedgerDigest(replay_digest(complete_records))
    assert digest.value == combined.hexdigest()
