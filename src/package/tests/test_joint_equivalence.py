from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.joint_equivalence import audit_corpus_file
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

BENCHMARK = Path(__file__).resolve().parents[1]
POLICY = BENCHMARK / "tests/fixtures/sim_to_real/approved_policy.yaml"
VALID = BENCHMARK / "tests/fixtures/sim_to_real/multi_pose_valid"
INVALID = BENCHMARK / "tests/fixtures/sim_to_real/single_pose_only"


def _canonical_digest(document: dict[str, object], *, omit: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != omit}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _copy_corpus(tmp_path: Path) -> Path:
    destination = tmp_path / "corpus"
    shutil.copytree(VALID, destination)
    return destination


def _document(root: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads((root / "corpus.json").read_text(encoding="utf-8")))


def _members(document: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", document["members"])


def _rewrite_manifest(root: Path, document: dict[str, object]) -> None:
    document["corpus_digest"] = _canonical_digest(document, omit="corpus_digest")
    (root / "corpus.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _member(root: Path, document: dict[str, object], index: int) -> dict[str, object]:
    path = root / cast("str", _members(document)[index]["path"])
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _rewrite_member(
    root: Path,
    document: dict[str, object],
    index: int,
    member: dict[str, object],
    *,
    update_hash: bool = True,
) -> None:
    entry = _members(document)[index]
    path = root / cast("str", entry["path"])
    path.write_text(json.dumps(member, indent=2) + "\n", encoding="utf-8")
    if update_hash:
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _rewrite_manifest(root, document)


def _expect_unproven(root: Path) -> None:
    with pytest.raises(RolloutViolation) as exc_info:
        audit_corpus_file(root, POLICY)
    assert exc_info.value.code is RolloutCode.EQUIVALENCE_UNPROVEN


def test_boolean_and_claimed_residual_only_corpus_is_unproven() -> None:
    _expect_unproven(INVALID)


def test_content_addressed_multi_pose_fixture_recomputes_and_audits() -> None:
    receipt = audit_corpus_file(VALID, POLICY)

    assert receipt["audited"] is True
    assert receipt["evidence_scope"] == "synthetic_test_fixture"
    assert receipt["genuine_physical_evidence"] is False
    assert receipt["deployment_valid"] is False
    assert receipt["policy_digest"] == (
        "54f41dcc964169459dc4d77f64bfa4f53bcc21e2d405931cd0eb51f41af11a6a"
    )
    assert receipt["digest"] == receipt["corpus_digest"]
    assert cast("int", receipt["fit_count"]) >= 11
    assert cast("int", receipt["held_out_count"]) >= 2
    assert cast("int", receipt["task_plane_pose_count"]) >= 2
    assert cast("float", receipt["max_fk_residual_m"]) <= 0.003
    assert receipt["computed_joint_order"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    assert len(cast("list[object]", receipt["computed_scales_rad_per_degree"])) == 5


@pytest.mark.parametrize("attack", ["order", "sign", "zero", "scale"])
def test_vector_mapping_attacks_are_unproven(tmp_path: Path, attack: str) -> None:
    root = _copy_corpus(tmp_path)
    document = _document(root)
    if attack == "order":
        order = cast("list[str]", document["simulator_joint_order"])
        order[0], order[1] = order[1], order[0]
        _rewrite_manifest(root, document)
    else:
        for index in range(len(_members(document))):
            member = _member(root, document, index)
            simulator = cast("dict[str, object]", member["simulator"])
            vector = cast("list[float]", simulator["joint_radians"])
            if attack == "sign":
                vector[0] = -vector[0]
            elif attack == "zero":
                vector[1] += 0.05
            else:
                vector[2] *= 1.1
            _rewrite_member(root, document, index, member)
    _expect_unproven(root)


def test_member_hash_attack_is_unproven(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    document = _document(root)
    member = _member(root, document, 0)
    member["claimed_fk_residual_m"] = 123.0
    _rewrite_member(root, document, 0, member, update_hash=False)
    _expect_unproven(root)


def test_timestamp_attack_is_unproven(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    document = _document(root)
    member = _member(root, document, 0)
    simulator = cast("dict[str, object]", member["simulator"])
    simulator["timestamp_s"] = cast("float", simulator["timestamp_s"]) + 1.0
    _rewrite_member(root, document, 0, member)
    _expect_unproven(root)


def test_held_out_attack_is_unproven(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    document = _document(root)
    for entry in _members(document):
        if entry["split"] == "held_out":
            entry["split"] = "fit"
    _rewrite_manifest(root, document)
    _expect_unproven(root)


def test_fk_attack_rejects_even_when_claimed_residual_is_small(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    document = _document(root)
    member = _member(root, document, -1)
    physical = cast("dict[str, object]", member["physical"])
    xyz = cast("list[float]", physical["measured_tool_xyz_m"])
    xyz[0] += 0.02
    member["claimed_fk_residual_m"] = 0.0
    _rewrite_member(root, document, -1, member)
    _expect_unproven(root)


def test_missing_real_corpus_cli_emits_truthful_blocker_without_output(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "scripts/audit_joint_equivalence_read_only.py"),
            "--synthetic-fixture",
            "--corpus",
            str(tmp_path / "absent-real-corpus"),
            "--policy",
            str(POLICY),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "EQUIVALENCE_UNPROVEN" in completed.stderr
    assert "genuine physical corpus is absent" in completed.stderr
    assert not output.exists()
