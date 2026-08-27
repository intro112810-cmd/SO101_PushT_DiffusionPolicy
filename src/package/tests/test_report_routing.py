from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from so101_pusht_benchmark.workspace import (
    WorkspacePolicyError,
    load_workspace_policy,
    runtime_artifact_root,
    validate_report_path,
)


def test_runtime_artifacts_route_only_to_experiment_root() -> None:
    policy = load_workspace_policy()
    workspace = policy["workspace"]
    routing = policy["report_routing"]
    assert routing["runtime_artifacts_root"] == workspace["artifact_root"]
    assert runtime_artifact_root().as_posix().endswith("/04_experiments/so101_pusht_benchmark")


def test_report_path_policy_rejects_project_root_and_allows_artifact_root() -> None:
    assert validate_report_path(runtime_artifact_root() / "result.json")
    with pytest.raises(WorkspacePolicyError):
        validate_report_path(runtime_artifact_root().parents[2] / "03_code/report.json")


def test_obsidian_markdown_exception_is_explicit_and_not_automatic() -> None:
    routing = load_workspace_policy()["report_routing"]
    assert routing["obsidian_exception_requires_user_request"] is True
    assert routing["no_automatic_obsidian_write"] is True
    assert routing["obsidian_markdown_exception"].endswith("/AI_보고서")
    with pytest.raises(WorkspacePolicyError):
        validate_report_path(Path(routing["obsidian_markdown_exception"]) / "report.md")
    assert validate_report_path(
        Path(routing["obsidian_markdown_exception"]) / "report.md",
        user_requested_obsidian=True,
    )


def test_artifact_symlink_components_and_targets_are_rejected(tmp_path: Path) -> None:
    artifact_root_patch = patch(
        "so101_pusht_benchmark.workspace.runtime_artifact_root", return_value=tmp_path
    )
    artifact_root_patch.start()
    real_dir = tmp_path / "real-dir"
    real_dir.mkdir()
    (real_dir / "target.json").write_text("data", encoding="utf-8")
    (tmp_path / "linked-dir").symlink_to(real_dir, target_is_directory=True)
    (tmp_path / "linked-file.json").symlink_to(real_dir / "target.json")
    try:
        with pytest.raises(WorkspacePolicyError):
            validate_report_path(tmp_path / "linked-dir" / "new.json")
        with pytest.raises(WorkspacePolicyError):
            validate_report_path(tmp_path / "linked-file.json")
    finally:
        artifact_root_patch.stop()


def test_existing_special_files_and_unsafe_parents_are_rejected(tmp_path: Path) -> None:
    artifact_root_patch = patch(
        "so101_pusht_benchmark.workspace.runtime_artifact_root", return_value=tmp_path
    )
    artifact_root_patch.start()
    fifo = tmp_path / "result.fifo"
    os.mkfifo(fifo)
    (tmp_path / "regular.json").write_text("data", encoding="utf-8")
    try:
        assert validate_report_path(tmp_path / "regular.json") == tmp_path / "regular.json"
        assert validate_report_path(tmp_path / "future.json") == tmp_path / "future.json"
        with pytest.raises(WorkspacePolicyError):
            validate_report_path(fifo)
        with pytest.raises(WorkspacePolicyError):
            validate_report_path(tmp_path / "regular.json" / "child.json")
    finally:
        artifact_root_patch.stop()


def test_obsidian_exception_is_flat_direct_markdown_only() -> None:
    root = Path(load_workspace_policy()["report_routing"]["obsidian_markdown_exception"])
    assert validate_report_path(root / "flat.md", user_requested_obsidian=True) == root / "flat.md"
    with pytest.raises(WorkspacePolicyError):
        validate_report_path(root / "nested" / "report.md", user_requested_obsidian=True)
    with pytest.raises(WorkspacePolicyError):
        validate_report_path(root / "report.txt", user_requested_obsidian=True)
