from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest
import yaml

PACKAGE_ROOT = Path(__file__).parents[1]


def _config(path: Path, target: int) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "pusht-so100-experiment-v1",
                "target_episode_count": target,
                "split_ratios": {"train": "0.8", "validation": "0.1", "test": "0.1"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _metadata(path: Path, accepted: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "total_episodes": accepted,
                "total_frames": accepted * 3,
            }
        ),
        encoding="utf-8",
    )
    return path


def _plan(metadata: Path, config: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["SO101_PUSHT_F710_DEVICES_JSON"] = "[]"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "so101_pusht_benchmark.cli",
            "freeze-experiment",
            "--metadata",
            str(metadata),
            "--experiment-config",
            str(config),
            "--dry-run",
        ],
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("target", "counts"),
    [
        (10, {"train": 8, "validation": 1, "test": 1}),
        (50, {"train": 40, "validation": 5, "test": 5}),
        (200, {"train": 160, "validation": 20, "test": 20}),
    ],
)
def test_target_met_metadata_reaches_artifact_free_manifest_plan(
    tmp_path: Path, target: int, counts: dict[str, int]
) -> None:
    config = _config(tmp_path / "experiment.yaml", target)
    metadata = _metadata(tmp_path / "info.json", target)

    result = _plan(metadata, config)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = cast("dict[str, object]", json.loads(result.stdout))
    assert payload["status"] == "dry-run-manifest-planned"
    assert payload["artifacts_created"] is False
    assert payload["accepted_episode_count"] == target
    assert payload["target_episode_count"] == target
    assert payload["remaining_episode_count"] == 0
    assert payload["split_counts"] == counts
    assert len(cast("list[str]", payload["selected_episode_ids"])) == target
    assert set(tmp_path.iterdir()) == {config, metadata}


@pytest.mark.parametrize("target", [10, 50, 200])
def test_zero_and_target_minus_one_fail_with_exact_progress_and_no_artifacts(
    tmp_path: Path, target: int
) -> None:
    config = _config(tmp_path / f"experiment-{target}.yaml", target)
    for accepted, missing in ((0, target), (target - 1, 1)):
        case = tmp_path / f"case-{accepted}"
        case.mkdir()
        metadata = _metadata(case / "info.json", accepted)

        result = _plan(metadata, config)

        assert result.returncode == 1
        assert result.stdout == (
            f"FAIL CLOSED: accepted episodes {accepted}/{target}; collect {missing} more\n"
        )
        assert result.stderr == ""
        assert set(case.iterdir()) == {metadata}
        assert not list(case.rglob("*.ckpt"))
        assert not list(case.rglob("comparison.json"))
        assert not list(case.rglob("comparison.md"))


def test_operator_dry_run_never_claims_existing_stage_completion(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.yaml", 10).resolve()

    result = subprocess.run(
        ["bash", "scripts/production_operator.sh", "--dry-run", str(config)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESUME: final evaluation complete" not in result.stdout
    assert "RESUME: validated" not in result.stdout
    assert (
        result.stdout.count(
            "DRY_RUN_NOTE: final evaluation completion will be claimed only after validation"
        )
        == 4
    )
    assert result.stdout.count("validate-production-artifact --stage training") == 4
    assert result.stdout.count("validate-production-artifact --stage bundle") == 4
    assert result.stdout.count("validate-production-artifact --stage evaluation") == 4


def test_operator_resume_routes_existing_report_through_read_only_validation() -> None:
    script = (PACKAGE_ROOT / "scripts/production_operator.sh").read_text(encoding="utf-8")

    assert 'if [[ "$MODE" == "--dry-run" || ! -e "$REPORT_OUTPUT" ]]; then' in script
    resume = script.split('echo "RESUME: validating existing final report', maxsplit=1)[1]
    assert "--validate-existing" in resume
    assert "validated and reused byte-identical final report" in resume
    assert "existing final report was preserved and was not reused" in resume
    assert "move it to a new quarantine path" in resume
    assert "rm " not in resume
    assert "preserving existing final report" not in script
    assert "preserving existing imported store" not in script
    assert "RESUME: completed non-final smoke" not in script


def test_bad_metadata_path_and_contract_fail_without_output(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.yaml", 10)
    missing = tmp_path / "missing-info.json"
    result = _plan(missing, config)
    assert result.returncode == 1
    assert result.stdout == (
        f"FAIL CLOSED: native collection metadata is not a regular file: {missing}\n"
    )

    malformed = tmp_path / "info.json"
    malformed.write_text('{"codebase_version":"v3.0","fps":10,"total_episodes":true}')
    result = _plan(malformed, config)
    assert result.returncode == 1
    assert result.stdout == (
        "FAIL CLOSED: native collection metadata total_episodes must be a non-negative integer\n"
    )
    assert set(tmp_path.iterdir()) == {config, malformed}
