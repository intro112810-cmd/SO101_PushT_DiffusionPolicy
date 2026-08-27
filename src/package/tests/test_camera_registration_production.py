from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import cast

from fake_camera_production import (
    fake_production_trust_store,
    fake_signature,
)
import pytest

from so101_pusht_benchmark.sim_to_real.camera_audit_cli import run_camera_audit_cli
from so101_pusht_benchmark.sim_to_real.camera_registration import audit_production_corpus_file
from so101_pusht_benchmark.sim_to_real.receipt_routing import CANONICAL_ROLLOUT_ROOT
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame import registration_evidence_digest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/fixtures/sim_to_real/camera_registration_fake_production"
POLICY = CORPUS / "production_policy.yaml"
IDENTITY = CORPUS / "live-identity.json"
AUTHORITY = CORPUS / "corpus-authority.json"
SUPERSEDED_CAMERA_DIGEST = "7a7c874c52f06cee16a2328ba04e3e504664f5b2fb30bda3994abccd8b275a96"
CorpusMutation = Callable[[dict[str, object], Path], None]


def _failure_publisher(log: list[object]) -> Callable[[Path, dict[str, object], bool], None]:
    def publish(_path: Path, _receipt: dict[str, object], _production: bool) -> None:
        log.append(object())

    return publish


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _resign_authority(root: Path, corpus: dict[str, object]) -> None:
    authority = _json(root / "corpus-authority.json")
    authority["corpus_digest"] = corpus["camera_digest"]
    content = {key: value for key, value in authority.items() if key != "binding_signature"}
    authority["binding_signature"] = fake_signature(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )
    (root / "corpus-authority.json").write_text(json.dumps(authority, indent=2) + "\n")


def _mutated(tmp_path: Path, mutation: CorpusMutation, *, resign: bool = True) -> Path:
    target = tmp_path / "production-corpus"
    shutil.copytree(CORPUS, target)
    corpus = _json(target / "corpus.json")
    mutation(corpus, target)
    corpus["camera_digest"] = registration_evidence_digest(corpus)
    (target / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n")
    if resign:
        _resign_authority(target, corpus)
    return target


def _physical_argv(
    output: Path,
    *,
    corpus: Path = CORPUS,
    policy: Path = POLICY,
    identity: Path = IDENTITY,
    authority: Path = AUTHORITY,
) -> list[str]:
    return [
        "--mode",
        "physical",
        "--corpus",
        str(corpus),
        "--policy",
        str(policy),
        "--identity",
        str(identity),
        "--corpus-authority",
        str(authority),
        "--output",
        str(output),
    ]


def _audit(path: Path = CORPUS) -> dict[str, object]:
    return audit_production_corpus_file(
        path,
        path / "production_policy.yaml",
        identity_path=path / "live-identity.json",
        authority_path=path / "corpus-authority.json",
        trust_store=fake_production_trust_store(),
    )


def test_signed_fake_production_raw_corpus_recomputes_and_derives_scope() -> None:
    receipt = _audit()

    assert receipt["audited"] is True
    assert receipt["evidence_scope"] == "authorized_physical_diagnostic"
    assert receipt["source_evidence_scope"] == "production"
    assert receipt["genuine_physical_corpus"] is True
    assert receipt["metrics_source"] == "recomputed_from_raw_points_and_matrices"
    assert receipt["policy_authority"] == "ProductionApprovedSafetyPolicy"
    assert receipt["identity_digest"] == _json(IDENTITY)["identity_digest"]
    assert receipt["corpus_authority_approval_id"] == "fake-camera-corpus-approval-v1"


def test_physical_cli_happy_publishes_only_to_canonical_lexical_route() -> None:
    published: list[tuple[Path, dict[str, object], bool]] = []
    output = CANONICAL_ROLLOUT_ROOT / "camera/fake-production-camera.json"
    result = run_camera_audit_cli(
        _physical_argv(output),
        production_trust_store=fake_production_trust_store(),
        publisher=lambda path, receipt, production: published.append((path, receipt, production)),
    )

    assert result == 0
    assert published == [(output, published[0][1], True)]
    assert published[0][1]["evidence_scope"] == "authorized_physical_diagnostic"


def test_noncanonical_physical_output_fails_before_input_io(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published: list[object] = []
    result = run_camera_audit_cli(
        _physical_argv(
            tmp_path / "noncanonical.json",
            corpus=tmp_path / "absent-corpus",
            policy=tmp_path / "absent-policy",
            identity=tmp_path / "absent-identity",
            authority=tmp_path / "absent-authority",
        ),
        production_trust_store=fake_production_trust_store(),
        publisher=_failure_publisher(published),
    )

    assert result == 2
    assert "canonical rollout root" in capsys.readouterr().err
    assert published == []


def test_missing_trust_store_or_corpus_publishes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published: list[object] = []
    output = CANONICAL_ROLLOUT_ROOT / "camera/missing-production-input.json"
    no_trust = run_camera_audit_cli(
        _physical_argv(output),
        publisher=_failure_publisher(published),
    )
    assert no_trust == 2
    assert "trust store" in capsys.readouterr().err

    no_corpus = run_camera_audit_cli(
        _physical_argv(output, corpus=tmp_path / "absent-corpus"),
        production_trust_store=fake_production_trust_store(),
        publisher=_failure_publisher(published),
    )
    assert no_corpus == 2
    assert "camera corpus is absent" in capsys.readouterr().err

    no_policy = run_camera_audit_cli(
        _physical_argv(output, policy=tmp_path / "absent-policy"),
        production_trust_store=fake_production_trust_store(),
        publisher=_failure_publisher(published),
    )
    assert no_policy == 2
    assert "policy" in capsys.readouterr().err.lower()
    assert published == []


def test_untrusted_production_policy_fails_before_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = POLICY.read_text(encoding="utf-8").replace(
        "binding_signature:",
        "binding_signature: 00 #",
        1,
    )
    policy = tmp_path / "untrusted-policy.yaml"
    policy.write_text(raw, encoding="utf-8")
    published: list[object] = []
    result = run_camera_audit_cli(
        _physical_argv(
            CANONICAL_ROLLOUT_ROOT / "camera/untrusted-policy.json",
            policy=policy,
        ),
        production_trust_store=fake_production_trust_store(),
        publisher=_failure_publisher(published),
    )
    assert result == 2
    assert "policy" in capsys.readouterr().err.lower()
    assert published == []


@pytest.mark.parametrize("scope", ["synthetic_test_fixture", "authorized_physical_diagnostic"])
def test_fixture_or_receipt_scope_string_cannot_masquerade_as_physical(
    tmp_path: Path, scope: str
) -> None:
    path = _mutated(
        tmp_path,
        lambda corpus, _root: corpus.__setitem__("evidence_scope", scope),
    )
    with pytest.raises(RolloutViolation) as caught:
        _audit(path)
    assert caught.value.code is RolloutCode.CAMERA_UNREGISTERED


def test_raw_member_tamper_rejects_after_valid_owner_signatures(tmp_path: Path) -> None:
    def mutate(_corpus: dict[str, object], root: Path) -> None:
        (root / "members/held-a.png").write_bytes(b"tampered")

    with pytest.raises(RolloutViolation) as caught:
        _audit(_mutated(tmp_path, mutate))
    assert caught.value.code is RolloutCode.CAMERA_UNREGISTERED


def test_provider_identity_drift_rejects(tmp_path: Path) -> None:
    path = _mutated(tmp_path, lambda _corpus, _root: None)
    identity = _json(path / "live-identity.json")
    identity["provider_digest"] = "9" * 64
    (path / "live-identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    with pytest.raises(RolloutViolation) as caught:
        _audit(path)
    assert caught.value.code in {RolloutCode.R_HASH_MISMATCH, RolloutCode.R_POLICY_UNAUTHORIZED}


def test_corpus_authority_signature_tamper_rejects(tmp_path: Path) -> None:
    path = _mutated(tmp_path, lambda _corpus, _root: None)
    authority = _json(path / "corpus-authority.json")
    authority["binding_signature"] = "0" * 64
    (path / "corpus-authority.json").write_text(json.dumps(authority, indent=2) + "\n")

    with pytest.raises(RolloutViolation) as caught:
        _audit(path)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_superseded_camera_digest_is_not_a_valid_production_identity() -> None:
    assert _json(AUTHORITY)["corpus_digest"] != SUPERSEDED_CAMERA_DIGEST
