from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.camera_registration import (
    audit_camera_registration,
    audit_corpus_file,
)
from so101_pusht_benchmark.sim_to_real.policy_types import CameraPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame import registration_evidence_digest

BENCHMARK = Path(__file__).resolve().parents[1]
POLICY = BENCHMARK / "tests/fixtures/sim_to_real/approved_policy.yaml"
VALID = BENCHMARK / "tests/fixtures/sim_to_real/camera_registration_valid"
INVALID = BENCHMARK / "tests/fixtures/sim_to_real/visual_only_alignment"
SCRIPT = BENCHMARK / "scripts/audit_camera_registration.py"
CorpusMutation = Callable[[dict[str, object], Path], None]


def _corpus(path: Path = VALID) -> dict[str, object]:
    raw = json.loads((path / "corpus.json").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _list(raw: object) -> list[object]:
    assert isinstance(raw, list)
    return cast("list[object]", raw)


def _mapping(raw: object) -> dict[str, object]:
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _float(raw: object) -> float:
    assert isinstance(raw, int | float)
    assert not isinstance(raw, bool)
    return float(raw)


def _mutated_corpus(tmp_path: Path, mutation: CorpusMutation, *, rehash: bool = True) -> Path:
    target = tmp_path / "corpus"
    shutil.copytree(VALID, target)
    corpus = _corpus(target)
    mutation(corpus, target)
    if rehash:
        corpus["camera_digest"] = registration_evidence_digest(corpus)
    (target / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    return target


def _rejects(path: Path) -> None:
    with pytest.raises(RolloutViolation) as exc_info:
        audit_corpus_file(path, POLICY)
    assert exc_info.value.code is RolloutCode.CAMERA_UNREGISTERED


def test_scalar_count_presence_only_corpus_is_rejected() -> None:
    scalar_only: dict[str, object] = {
        "intrinsics": [600.0, 0.0, 200.0, 0.0, 600.0, 200.0, 0.0, 0.0, 1.0],
        "distortion": [0.0] * 5,
        "camera_to_table": [1.0] * 12,
        "physical_to_sim_se2": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "held_out_reprojection_error_px": 0.0,
        "held_out_correspondences": 999,
        "device_hash": "a" * 64,
        "resolution": [400, 400],
        "crop": [0, 0, 400, 400],
        "orientation_hash": "b" * 64,
        "config_hash": "c" * 64,
    }
    with pytest.raises(RolloutViolation) as exc_info:
        audit_camera_registration(
            scalar_only,
            corpus_root=VALID,
            source_scope="synthetic_test_fixture",
            thresholds=CameraPolicy(1.5, 12, 2.0),
        )
    assert exc_info.value.code is RolloutCode.CAMERA_UNREGISTERED


def test_synthetic_raw_geometry_recomputes_and_audits() -> None:
    receipt = audit_corpus_file(VALID, POLICY)
    assert receipt["audited"] is True
    assert receipt["evidence_scope"] == "synthetic_test_fixture"
    assert receipt["metrics_source"] == "recomputed_from_raw_points_and_matrices"
    assert receipt["fit_correspondences"] == 12
    assert receipt["held_out_correspondences"] == 12
    assert _float(receipt["held_out_reprojection_error_px"]) < 1e-9
    assert _float(receipt["max_correspondence_error_px"]) < 1e-9
    assert _float(receipt["held_out_physical_to_sim_residual_m"]) < 1e-12
    member_digests = receipt["member_digests"]
    assert isinstance(member_digests, dict)
    typed_member_digests = cast("dict[object, object]", member_digests)
    assert len(typed_member_digests) == 4
    assert isinstance(receipt["digest"], str)
    assert len(str(receipt["digest"])) == 64


def test_visual_only_alignment_is_unregistered() -> None:
    _rejects(INVALID)


def _mutate_image(_corpus: dict[str, object], root: Path) -> None:
    (root / "members/fit-a.png").write_bytes(b"mutated-image")


MUTATIONS: tuple[CorpusMutation, ...] = (
    _mutate_image,
    lambda corpus, _root: _mapping(_list(corpus["members"])[0]).__setitem__("sha256", "0" * 64),
    lambda corpus, _root: _mapping(_list(corpus["fit_correspondences"])[0]).__setitem__(
        "image_point_px", [0.0, 0.0]
    ),
    lambda corpus, _root: _mapping(_list(corpus["fit_correspondences"])[0]).__setitem__(
        "table_point_m", [0.03, 0.02, 0.0]
    ),
    lambda corpus, _root: _mapping(_list(corpus["fit_correspondences"])[0]).__setitem__(
        "simulation_point_m", [0.5, 0.5]
    ),
    lambda corpus, _root: _mapping(corpus["intrinsics"]).__setitem__(
        "matrix", [650.0, 0.0, 200.0, 0.0, 600.0, 200.0, 0.0, 0.0, 1.0]
    ),
    lambda corpus, _root: _mapping(corpus["distortion"]).__setitem__(
        "coefficients", [0.2, -0.005, 0.0005, -0.0003, 0.001]
    ),
    lambda corpus, _root: _mapping(corpus["camera_to_table"]).__setitem__(
        "matrix",
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.65, 0.0, 0.0, 0.0, 1.0],
    ),
    lambda corpus, _root: _mapping(corpus["camera_to_table"]).__setitem__(
        "direction", "table_to_camera"
    ),
    lambda corpus, _root: _mapping(corpus["physical_to_sim"]).__setitem__(
        "matrix_2x3", [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    ),
    lambda corpus, _root: _mapping(corpus["physical_to_sim"]).__setitem__(
        "physical_units", "millimeters"
    ),
    lambda corpus, _root: corpus.__setitem__("camera_digest", "0" * 64),
    lambda corpus, _root: corpus.__setitem__(
        "held_out_correspondences", list(_list(corpus["fit_correspondences"]))
    ),
    lambda corpus, _root: corpus.__setitem__(
        "checkpoint_view_members", [_list(corpus["checkpoint_view_members"])[0]]
    ),
    lambda corpus, _root: _mapping(_list(corpus["checkpoint_view_members"])[0]).__setitem__(
        "timestamp", "2026-08-23T12:00:00.000Z"
    ),
)
MUTATION_IDS = (
    "raw-image",
    "member-digest",
    "image-point",
    "table-point",
    "simulation-point",
    "intrinsics-matrix",
    "distortion",
    "extrinsics-matrix",
    "frame-direction",
    "physical-simulation-matrix",
    "units",
    "corpus-hash",
    "heldout-reuse",
    "checkpoint-coverage",
    "checkpoint-timestamp",
)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=MUTATION_IDS)
def test_raw_evidence_mutation_attacks_reject(tmp_path: Path, mutation: CorpusMutation) -> None:
    path = _mutated_corpus(tmp_path, mutation, rehash=mutation is not MUTATIONS[11])
    _rejects(path)


def test_degenerate_geometry_rejects(tmp_path: Path) -> None:
    def collapse(corpus: dict[str, object], _root: Path) -> None:
        for item in _list(corpus["fit_correspondences"]):
            correspondence = _mapping(item)
            correspondence["table_point_m"] = [0.0, 0.0, 0.0]
            correspondence["simulation_point_m"] = [0.0, 0.0]
            correspondence["image_point_px"] = [200.0, 200.0]

    _rejects(_mutated_corpus(tmp_path, collapse))


def test_per_correspondence_policy_limit_is_recomputed(tmp_path: Path) -> None:
    def move_one(corpus: dict[str, object], _root: Path) -> None:
        point = _mapping(_list(corpus["held_out_correspondences"])[0])
        observed = _list(point["image_point_px"])
        point["image_point_px"] = [_float(observed[0]) + 2.1, _float(observed[1])]

    _rejects(_mutated_corpus(tmp_path, move_one))


def test_cli_happy_path_marks_fixture_without_production_masquerade(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "fixture",
            "--corpus",
            str(VALID),
            "--policy",
            str(POLICY),
            "--output",
            str(output),
        ],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["evidence_scope"] == "synthetic_test_fixture"
    assert receipt["policy_evidence"] == "fixture_policy_not_production_authority"


def test_missing_real_corpus_cli_emits_truthful_blocker_without_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "fixture",
            "--corpus",
            str(tmp_path / "absent-real-corpus"),
            "--policy",
            str(POLICY),
            "--output",
            str(output),
        ],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "camera corpus is absent" in result.stderr
    assert "synthetic" in result.stderr
    assert not output.exists()
