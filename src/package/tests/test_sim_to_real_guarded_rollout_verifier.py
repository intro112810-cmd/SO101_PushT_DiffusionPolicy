"""Non-actuating verification of complete guarded-rollout session receipts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

BENCHMARK = Path(__file__).resolve().parents[1]
SCRIPT = BENCHMARK / "scripts/verify_guarded_rollout.py"
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
COMPLETE = FIXTURES / "complete_session"
TAMPERED = FIXTURES / "tampered_ack_session"


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run(session: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--session", str(session), *extra],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _rewrite_manifest_hash(session: Path, member: str) -> None:
    manifest_path = session / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][member] = hashlib.sha256((session / member).read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_complete_session_is_accepted_without_mutation() -> None:
    before = _tree_digest(COMPLETE)
    result = _run(COMPLETE, "--verify-cleanup")
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["valid"] is True
    assert receipt["session_id"] == "fixture-complete-session"
    assert receipt["ledger_record_count"] == 13
    assert receipt["single_step_write_count"] == 1
    assert receipt["bounded_write_count"] == 3
    assert receipt["cleanup_verified"] is True
    assert receipt["actuation_performed_by_verifier"] is False
    assert _tree_digest(COMPLETE) == before


def test_tampered_ack_session_is_rejected_without_mutation() -> None:
    before = _tree_digest(TAMPERED)
    result = _run(TAMPERED, "--verify-cleanup")
    assert result.returncode == 2
    assert "R_HASH_MISMATCH" in result.stderr
    assert result.stdout == ""
    assert _tree_digest(TAMPERED) == before


def test_cleanup_claim_with_remaining_temp_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    shutil.copytree(COMPLETE, session)
    cleanup_path = session / "cleanup.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["temporary_paths_remaining"] = [str(tmp_path / "stale-guarded-rollout")]
    cleanup_path.write_text(json.dumps(cleanup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_hash(session, "cleanup.json")

    result = _run(session, "--verify-cleanup")
    assert result.returncode == 2
    assert "cleanup" in result.stderr.lower()
    assert not (session / "verification_receipt.json").exists()


def test_verifier_source_has_no_writer_or_hardware_capability() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert not any("writer" in name or "lerobot" in name for name in imported)
    for forbidden in ("sync_write", "Goal_Position", "send_action", "SOFollower", "sleep("):
        assert forbidden not in source
