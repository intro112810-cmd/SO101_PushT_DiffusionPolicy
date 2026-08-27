"""Dry-run CLI exposes the authentic-history contract without tracebacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import NoReturn

import pytest

from scripts import run_sim_to_real_dry_run as dry_run
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    CANONICAL_ROLLOUT_ROOT,
    ReceiptPathIdentity,
)

BENCHMARK = Path(__file__).resolve().parents[1]
SCRIPT = BENCHMARK / "scripts/run_sim_to_real_dry_run.py"


def _write_follower(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "mode": "read_only_follower_state",
                "positions_degrees": {
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 0.0,
                    "gripper": 0.0,
                },
                "raw_encoder": {},
                "motor_writes_performed": False,
                "actuation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_dry_run_help_names_authentic_history_inputs() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "--samples" in result.stdout
    assert "--lineage" in result.stdout
    assert "--joint" in result.stdout
    assert "--camera" in result.stdout
    assert "Traceback" not in result.stderr


def test_production_dry_run_preserves_lexical_alias_until_verified_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io_root = tmp_path / "verified-resolved-io"
    io_root.mkdir()
    lexical_output = CANONICAL_ROLLOUT_ROOT / "sessions/dry-run-lexical"
    follower = _write_follower(tmp_path / "follower.json")

    def locate(path: Path) -> ReceiptPathIdentity:
        lexical = path if path.is_absolute() else Path.cwd() / path
        relative = lexical.relative_to(lexical_output)
        resolved = io_root / relative
        return ReceiptPathIdentity(lexical, resolved, True)

    def fake_shadow(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        assert output == lexical_output / "shadow_inference.json"
        (io_root / "shadow_inference.json").write_text(
            json.dumps(
                {
                    "mode": "physical_frame_shadow_only",
                    "model": "dp_cnn",
                    "artifact_id": "local-dp_cnn-recovered-v3-seed0",
                    "evidence_scope": "production",
                    "policy_evidence": "authentic_frozen_production",
                    "frame_sha256": "a" * 64,
                    "checkpoint_image_contract": "CCW90 RGB uint8[96,96,3]",
                    "agent_pos": [0.0] * 5,
                    "agent_pos_source": "receipt_bound_affine_mapping",
                    "predicted_actions": [[0.0, 0.0] for _ in range(8)],
                    "deployment_valid": False,
                    "actuation_performed": False,
                    "follower_motor_writes_performed": False,
                    "follower_actuation_performed": False,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    def validate(identity: ReceiptPathIdentity, *, production: bool) -> ReceiptPathIdentity:
        assert production is True
        return identity

    rendered: list[tuple[Path, Path, str]] = []

    def fake_preview(
        _actions: object,
        *,
        png_path: Path,
        mp4_path: Path,
        seed: int,
        evidence_scope: str,
    ) -> dict[str, object]:
        assert seed == 0
        rendered.append((png_path, mp4_path, evidence_scope))
        png_path.write_bytes(b"png")
        mp4_path.write_bytes(b"mp4")
        return {"evidence_scope": evidence_scope, "deployment_valid": False}

    monkeypatch.setattr(dry_run, "locate_receipt_path", locate)
    monkeypatch.setattr(dry_run, "validate_receipt_identity", validate)
    monkeypatch.setattr(dry_run.subprocess, "run", fake_shadow)
    monkeypatch.setattr(dry_run, "render_prediction_preview", fake_preview)
    args = argparse.Namespace(
        artifact_root=tmp_path,
        model="dp_cnn",
        artifact=None,
        frame=tmp_path / "frame.png",
        samples=tmp_path / "samples.json",
        lineage=tmp_path / "lineage.json",
        joint=tmp_path / "joint.json",
        camera=tmp_path / "camera.json",
        follower_state=follower,
        output_dir=lexical_output,
        policy_seed=0,
        preview_seed=0,
        preview_mp4=lexical_output / "preview.mp4",
        preview_png=lexical_output / "preview.png",
    )

    receipt = dry_run.run_dry_run(args)
    assert receipt["evidence_scope"] == "production"
    assert (io_root / "dry_run_receipt.json").is_file()
    assert rendered == [(io_root / "preview.png", io_root / "preview.mp4", "production")]
    assert not (lexical_output / "dry_run_receipt.json").exists()


@pytest.mark.parametrize(
    "attack", ["outside", "resolved", "traversal", "parent-symlink", "leaf-symlink"]
)
def test_production_preview_path_attacks_reject_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    lexical_output = CANONICAL_ROLLOUT_ROOT / "sessions/preview-routing-red"
    follower = _write_follower(tmp_path / "follower.json")
    if attack == "outside":
        png = tmp_path / "OUTSIDE.png"
        mp4 = tmp_path / "OUTSIDE.mp4"
    elif attack == "resolved":
        resolved = Path(
            "/data/df/02_InTro_Project/04_experiments/so101_pusht_benchmark/"
            "inference/sim_to_real_rollout/sessions/preview-routing-red"
        )
        png, mp4 = resolved / "preview.png", resolved / "preview.mp4"
    elif attack == "traversal":
        png = lexical_output / "nested/../OUTSIDE.png"
        mp4 = lexical_output / "nested/../OUTSIDE.mp4"
    elif attack == "parent-symlink":
        parent = tmp_path / "preview-parent"
        parent.symlink_to(tmp_path, target_is_directory=True)
        png, mp4 = parent / "preview.png", parent / "preview.mp4"
    else:
        png = tmp_path / "preview.png"
        mp4 = tmp_path / "preview.mp4"
        png.symlink_to(tmp_path / "foreign.png")
        mp4.symlink_to(tmp_path / "foreign.mp4")

    def fail_shadow(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("shadow provider ran before preview routing")

    def fail_preview(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("preview rendered before routing")

    monkeypatch.setattr(dry_run.subprocess, "run", fail_shadow)
    monkeypatch.setattr(dry_run, "render_prediction_preview", fail_preview)
    args = argparse.Namespace(
        artifact_root=tmp_path,
        model="dp_cnn",
        artifact=None,
        frame=tmp_path / "frame.png",
        samples=tmp_path / "samples.json",
        lineage=tmp_path / "lineage.json",
        joint=tmp_path / "joint.json",
        camera=tmp_path / "camera.json",
        follower_state=follower,
        output_dir=lexical_output,
        policy_seed=0,
        preview_seed=0,
        preview_mp4=mp4,
        preview_png=png,
    )

    with pytest.raises(ValueError, match=r"canonical|alias|traversal|symlink"):
        dry_run.run_dry_run(args)
    assert not png.exists()
    assert not mp4.exists()
    assert not lexical_output.exists()


def test_production_canonical_dry_run_missing_evidence_blocks_without_publication(
    tmp_path: Path,
) -> None:
    output = CANONICAL_ROLLOUT_ROOT / "sessions/missing-evidence-dry-run"
    result = _run(
        "--artifact-root",
        str(tmp_path),
        "--model",
        "dp_cnn",
        "--frame",
        str(tmp_path / "missing.png"),
        "--samples",
        str(tmp_path / "missing-samples.json"),
        "--lineage",
        str(tmp_path / "missing-lineage.json"),
        "--joint",
        str(tmp_path / "missing-joint.json"),
        "--camera",
        str(tmp_path / "missing-camera.json"),
        "--follower-state",
        str(tmp_path / "missing-follower.json"),
        "--output-dir",
        str(output),
    )

    assert result.returncode == 2
    assert "R_MISSING" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_dry_run_bad_input_returns_typed_error_without_traceback(tmp_path: Path) -> None:
    result = _run(
        "--artifact-root",
        str(BENCHMARK),
        "--model",
        "dp_cnn",
        "--frame",
        str(tmp_path / "missing.png"),
        "--samples",
        str(tmp_path / "missing-samples.json"),
        "--lineage",
        str(tmp_path / "missing-lineage.json"),
        "--joint",
        str(tmp_path / "missing-joint.json"),
        "--camera",
        str(tmp_path / "missing-camera.json"),
        "--follower-state",
        str(tmp_path / "missing-follower.json"),
        "--output-dir",
        str(tmp_path / "output"),
    )

    assert result.returncode == 2
    assert "R_MISSING" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "output").exists()
