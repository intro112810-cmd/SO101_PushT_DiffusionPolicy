"""Canonical durable routing and complete-session manifest regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    CANONICAL_ROLLOUT_ROOT,
    ReceiptRoutingError,
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_path,
)
from so101_pusht_benchmark.sim_to_real.session_manifest import (
    CANONICAL_SESSION_MEMBERS,
    manifest_digest,
    write_session_manifest,
)
from so101_pusht_benchmark.sim_to_real.session_verifier import verify_guarded_session
from so101_pusht_benchmark.sim_to_real.session_verifier_io import load_session
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation

BENCHMARK = Path(__file__).resolve().parents[1]
LEGACY_COMPLETE = BENCHMARK / "tests/fixtures/sim_to_real/complete_session"


def test_production_receipts_require_canonical_lexical_and_resolved_roots(
    tmp_path: Path,
) -> None:
    accepted = CANONICAL_ROLLOUT_ROOT / "sessions/production-001/lineage.json"
    identity = locate_receipt_path(accepted)
    assert identity.lexical == accepted
    assert identity.resolved == Path(
        "/data/df/02_InTro_Project/04_experiments/so101_pusht_benchmark/"
        "inference/sim_to_real_rollout/sessions/production-001/lineage.json"
    )
    assert identity.canonical is True
    assert validate_receipt_path(accepted, production=True) == accepted

    with pytest.raises(ReceiptRoutingError, match="canonical rollout root"):
        validate_receipt_path(tmp_path / "production-lineage.json", production=True)
    with pytest.raises(ReceiptRoutingError, match="without alias"):
        validate_receipt_path(identity.resolved, production=True)


def test_fixture_receipts_are_labeled_and_cannot_enter_canonical_root(tmp_path: Path) -> None:
    accepted = tmp_path / "fixture-session/raw_samples.json"
    assert validate_receipt_path(accepted, production=False) == accepted

    with pytest.raises(ReceiptRoutingError, match="fixture-only"):
        validate_receipt_path(
            CANONICAL_ROLLOUT_ROOT / "sessions/fake-production/raw_samples.json",
            production=False,
        )


def test_routing_rejects_traversal_and_nested_symlink(tmp_path: Path) -> None:
    with pytest.raises(ReceiptRoutingError, match="traversal"):
        validate_receipt_path(tmp_path / "session/../escape.json", production=False)

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ReceiptRoutingError, match="symlink"):
        prepare_receipt_directory(alias / "session", production=False)


def _stage_complete_session(root: Path) -> None:
    root.mkdir()
    for member in CANONICAL_SESSION_MEMBERS:
        source = LEGACY_COMPLETE / member
        if source.is_file():
            shutil.copy2(source, root / member)
        elif member.endswith(".jsonl"):
            (root / member).write_text(
                json.dumps({"evidence_scope": "test_fixture_only", "kind": member}) + "\n",
                encoding="utf-8",
            )
        elif member == "approved_policy.yaml":
            shutil.copy2(
                BENCHMARK / "tests/fixtures/sim_to_real/approved_policy.yaml",
                root / member,
            )
        elif member == "single_step_authorization.json":
            shutil.copy2(
                BENCHMARK / "tests/fixtures/sim_to_real/single_step_authorization.json",
                root / member,
            )
        elif member == "bounded_authorization.json":
            shutil.copy2(
                BENCHMARK / "tests/fixtures/sim_to_real/bounded_rollout_authorization.json",
                root / member,
            )
        else:
            (root / member).write_text(
                json.dumps(
                    {
                        "evidence_scope": "test_fixture_only",
                        "fixture_only": True,
                        "kind": member,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def test_v2_manifest_binds_exact_complete_fixture_members(tmp_path: Path) -> None:
    session = tmp_path / "fixture-session"
    _stage_complete_session(session)
    manifest = write_session_manifest(
        session,
        session_id="fixture-canonical-routing",
    )

    loaded = load_session(session)
    assert manifest == session / "session_manifest.json"
    assert loaded.session_id == "fixture-canonical-routing"
    assert loaded.evidence_scope == "test_fixture_only"
    assert frozenset(loaded.members) == CANONICAL_SESSION_MEMBERS
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(document["session_digest"]) == 64
    assert document["status"] == "COMPLETE"
    receipt = verify_guarded_session(session, verify_cleanup=True)
    assert receipt.valid is True
    assert receipt.cleanup_verified is True
    assert receipt.actuation_performed_by_verifier is False


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "symlink", "partial", "stale"])
def test_v2_manifest_rejects_incomplete_or_changed_members(tmp_path: Path, mutation: str) -> None:
    session = tmp_path / "fixture-session"
    _stage_complete_session(session)
    write_session_manifest(
        session,
        session_id="fixture-canonical-routing",
    )

    target = session / "raw_samples.json"
    if mutation == "missing":
        target.unlink()
    elif mutation == "unexpected":
        (session / "unexpected.partial").write_text("partial", encoding="utf-8")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(session / "lineage.json")
    elif mutation == "partial":
        manifest = session / "session_manifest.json"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["status"] = "PARTIAL"
        manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    else:
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    with pytest.raises(RolloutViolation):
        load_session(session)


def test_runbook_uses_literal_canonical_root_and_exact_manifest_cli() -> None:
    runbook = (BENCHMARK / "docs/GUARDED_SIM_TO_REAL_RUNBOOK_KO.md").read_text(encoding="utf-8")
    literal = (
        "/home/intro/InternLab/02_InTro_Project/04_experiments/"
        "so101_pusht_benchmark/inference/sim_to_real_rollout"
    )
    assert literal in runbook
    assert "scripts/build_guarded_session_manifest.py" in runbook
    assert "evidence_scope" in runbook
    assert "--evidence-scope" not in runbook
    assert all(member in runbook for member in CANONICAL_SESSION_MEMBERS)


def test_exact_22_member_fixture_cannot_masquerade_as_production(tmp_path: Path) -> None:
    session = tmp_path / "coherent-fixture-session"
    _stage_complete_session(session)
    manifest = write_session_manifest(session, session_id="fixture-masquerade-red")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["evidence_scope"] == "test_fixture_only"
    document["evidence_scope"] = "authorized_physical_diagnostic"
    document["session_digest"] = manifest_digest(document)
    manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RolloutViolation, match="derived evidence scope"):
        load_session(session)


def test_v2_manifest_rejects_traversal_member_even_when_rehashed(tmp_path: Path) -> None:
    session = tmp_path / "fixture-session"
    _stage_complete_session(session)
    manifest = write_session_manifest(
        session,
        session_id="fixture-canonical-routing",
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    record = document["files"].pop("raw_samples.json")
    document["files"]["../raw_samples.json"] = record
    manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RolloutViolation, match="unsafe session member"):
        load_session(session)
