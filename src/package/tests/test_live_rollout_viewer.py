from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import importlib.util


PACKAGE_ROOT = Path(__file__).parents[1]


def test_live_rollout_viewer_help_exposes_realtime_loop_option() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/live_rollout_viewer.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--viewer" in result.stdout
    assert "--viewer-3d" in result.stdout
    assert "--loop" in result.stdout


def test_feedback_artifact_generator_help_exposes_publish_asset_directory() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_feedback_artifacts.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--asset-dir" in result.stdout


def test_trajectory_montage_uses_every_evaluation_seed() -> None:
    script = PACKAGE_ROOT / "scripts/generate_trajectory_montage.py"
    specification = importlib.util.spec_from_file_location("trajectory_montage", script)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert module.montage_seeds() == tuple(range(100000, 100100))


def test_trajectory_montage_help_exposes_unique_output_name() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_trajectory_montage.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-name" in result.stdout


def test_case_video_generator_help_exposes_publish_asset_directory() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_case_videos.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--asset-dir" in result.stdout


def test_recovered_checkpoint_rollout_help_exposes_checkpoint_root() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/run_recovered_checkpoint_rollout.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint-root" in result.stdout


def test_recovered_checkpoint_selects_matching_legacy_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = PACKAGE_ROOT / "scripts/run_recovered_checkpoint_rollout.py"
    specification = importlib.util.spec_from_file_location("recovered_rollout", script)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.delenv("PUSHT_SINGLE_CAM", raising=False)
    monkeypatch.delenv("PUSHT_LOCAL_BUDGET", raising=False)

    module.configure_legacy_local_runtime(
        {
            "shape_meta": {
                "obs": {
                    "cam_top": {"shape": [3, 96, 96], "type": "rgb"},
                    "agent_pos": {"shape": [5], "type": "low_dim"},
                },
                "action": {"shape": [2]},
            }
        },
        optimizer_updates=400_000,
    )

    assert os.environ["PUSHT_SINGLE_CAM"] == "1"
    assert os.environ["PUSHT_LOCAL_BUDGET"] == "1"


def test_recovered_checkpoint_rollout_rejects_missing_receipt(tmp_path: Path) -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_recovered_checkpoint_rollout.py",
            "--checkpoint-root",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "unused.json"),
        ],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 2
    assert "missing training receipt" in result.stderr
    assert "Traceback" not in result.stderr


def test_layout_preview_generator_help_exposes_publish_asset_directory() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_layout_preview.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--asset-dir" in result.stdout


def test_three_view_walkthrough_generator_help_exposes_publish_asset_directory() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_three_view_walkthrough.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--asset-dir" in result.stdout


def test_outcome_three_view_generator_help_exposes_publish_asset_directory() -> None:
    environment = os.environ.copy()
    source_root = str(PACKAGE_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    )

    result = subprocess.run(
        [sys.executable, "scripts/generate_outcome_three_view_videos.py", "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--asset-dir" in result.stdout
