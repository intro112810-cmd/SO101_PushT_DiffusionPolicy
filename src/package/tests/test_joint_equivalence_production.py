from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.joint_equivalence import (
    audit_corpus_file,
    audit_production_corpus_file,
)
from so101_pusht_benchmark.sim_to_real.joint_equivalence_cli import (
    canonical_receipt_bytes,
    run_joint_equivalence_cli,
)
from so101_pusht_benchmark.sim_to_real.policy_approval import ProductionTrustStore
from so101_pusht_benchmark.sim_to_real.receipt_routing import CANONICAL_ROLLOUT_ROOT
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests/fixtures/sim_to_real/joint_equivalence_fake_production"
SYNTHETIC = ROOT / "tests/fixtures/sim_to_real/multi_pose_valid"
FIXTURE_POLICY = ROOT / "tests/fixtures/sim_to_real/approved_policy.yaml"
SCRIPT = ROOT / "scripts/audit_joint_equivalence_read_only.py"
_KEY = b"fake-joint-production-owner-anchor-v1"
_SIGNER = "fake-joint-production-owner@example.invalid"
_SCHEME = "hmac-sha256-fake-production-v1"


class _FakeOwnerAnchor:
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        expected = hmac.new(_KEY, content, hashlib.sha256).hexdigest()
        return (
            signer_id == _SIGNER
            and scheme == _SCHEME
            and hmac.compare_digest(expected, signature_hex)
        )


def _trust_store() -> ProductionTrustStore:
    return ProductionTrustStore.from_owner_anchors((_FakeOwnerAnchor(),))


@contextmanager
def _canonical_output() -> Generator[Path, None, None]:
    with TemporaryDirectory(prefix="pytest-joint-audit-", dir=CANONICAL_ROLLOUT_ROOT) as temporary:
        yield Path(temporary) / "joint-equivalence.json"


def _production_args(output: Path, corpus: Path = FAKE) -> list[str]:
    return [
        "--governed-physical",
        "--corpus",
        str(corpus),
        "--policy",
        str(FAKE / "production_policy.yaml"),
        "--corpus-authority",
        str(FAKE / "corpus-authority.json"),
        "--output",
        str(output),
    ]


def _rejects_production(corpus: Path, authority: Path | None = None) -> None:
    with pytest.raises(RolloutViolation):
        audit_production_corpus_file(
            corpus,
            FAKE / "production_policy.yaml",
            authority if authority is not None else corpus / "corpus-authority.json",
            trust_store=_trust_store(),
        )


def test_truthful_blocker_records_zero_genuine_corpora_and_exact_inputs() -> None:
    blocker = cast(
        "dict[str, object]",
        json.loads(
            (
                ROOT / "tests/fixtures/sim_to_real/joint-equivalence-genuine-blocker-v2.json"
            ).read_text(encoding="utf-8")
        ),
    )
    assert blocker["derived_v2_auditor_available"] is True
    assert blocker["genuine_physical_corpus_count"] == 0
    assert blocker["production_receipt_published"] is False
    inputs = cast("list[dict[str, object]]", blocker["external_inputs"])
    for item in inputs:
        path = ROOT.parent.parent / cast("str", item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["sufficient"] is False


def test_fake_signed_physical_corpus_recomputes_and_is_authorized() -> None:
    receipt = audit_production_corpus_file(
        FAKE,
        FAKE / "production_policy.yaml",
        FAKE / "corpus-authority.json",
        trust_store=_trust_store(),
    )

    assert receipt["audited"] is True
    assert receipt["evidence_scope"] == "authorized_physical_diagnostic"
    assert receipt["genuine_physical_evidence"] is True
    assert receipt["deployment_valid"] is True
    assert receipt["provider_digest"] == "1" * 64
    assert receipt["device_digest"] == "d" * 64
    assert receipt["calibration_digest"] == "e" * 64
    assert receipt["capture_id"] == "fake-read-only-capture-001"
    assert receipt["corpus_identity_digest"] == (
        "721c8ceecbd5b7063f20f6740f1046a06650cbdeaf7f04294bf4dc8b5fa7db83"
    )
    assert cast("float", receipt["max_fk_residual_m"]) <= 0.003


def test_production_cli_publishes_verified_canonical_bytes() -> None:
    with _canonical_output() as output:
        assert run_joint_equivalence_cli(_production_args(output), trust_store=_trust_store()) == 0
        encoded = output.read_bytes()
        document = cast("dict[str, object]", json.loads(encoded))
        assert encoded == canonical_receipt_bytes(document)
        assert document["evidence_scope"] == "authorized_physical_diagnostic"


def test_fixture_policy_and_scope_claim_cannot_masquerade_as_production() -> None:
    with pytest.raises(RolloutViolation) as caught:
        audit_production_corpus_file(
            FAKE,
            FIXTURE_POLICY,
            FAKE / "corpus-authority.json",
            trust_store=_trust_store(),
        )
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED

    with pytest.raises(RolloutViolation):
        audit_corpus_file(FAKE, FAKE / "production_policy.yaml")

    with _canonical_output() as output:
        assert (
            run_joint_equivalence_cli(
                _production_args(output, SYNTHETIC), trust_store=_trust_store()
            )
            == 2
        )
        assert not output.exists()


def test_default_process_has_no_production_trust_and_publishes_nothing() -> None:
    with _canonical_output() as output:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *_production_args(output)],
            cwd=ROOT,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")},
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 2
        assert "trust store is unavailable" in completed.stderr
        assert not output.exists()


@pytest.mark.parametrize("attack", ["signature", "provider", "member", "identity"])
def test_production_authority_and_evidence_tampering_rejects(tmp_path: Path, attack: str) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FAKE, corpus)
    authority_path = corpus / "corpus-authority.json"
    authority = cast("dict[str, object]", json.loads(authority_path.read_text(encoding="utf-8")))
    if attack == "signature":
        authority["binding_signature"] = "0" * 64
        authority_path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")
    elif attack == "provider":
        authority["provider_digest"] = "2" * 64
        authority_path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")
    elif attack == "member":
        member = corpus / "members/held-task-b.json"
        member.write_text(member.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        manifest_path = corpus / "corpus.json"
        manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
        bindings = cast("dict[str, object]", manifest["production_bindings"])
        bindings["device_digest"] = "a" * 64
        manifest.pop("corpus_digest")
        manifest["corpus_digest"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _rejects_production(corpus, authority_path)


def test_noncanonical_existing_output_and_noncanonical_route_fail_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    assert run_joint_equivalence_cli(_production_args(outside), trust_store=_trust_store()) == 2
    assert not outside.exists()

    with _canonical_output() as output:
        prior = b'{"noncanonical": true}\n'
        output.write_bytes(prior)
        assert run_joint_equivalence_cli(_production_args(output), trust_store=_trust_store()) == 2
        assert output.read_bytes() == prior
