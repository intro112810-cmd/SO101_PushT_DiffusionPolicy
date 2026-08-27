from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from so101_pusht_benchmark.native_cli import command_parser

PACKAGE_ROOT = Path(__file__).parents[1]
ACTIVE_COMMANDS = (
    "validate-contract",
    "inspect-env",
    "collect-native",
    "import-native",
    "export-native",
    "freeze-experiment",
    "train-model",
    "export-inference-bundle",
    "evaluate-model",
    "compare-models",
    "native-env-smoke",
)


def _help(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "so101_pusht_benchmark.cli", *arguments, "--help"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def test_top_level_help_exposes_only_active_native_route() -> None:
    result = _help()
    assert result.returncode == 0
    for command in ACTIVE_COMMANDS:
        assert command in result.stdout
    for inactive in (
        "collect-sim",
        "import-repo-store",
        "export-paper-view",
        "probe-gamepad",
        "validate-pilot",
        "replay-episode",
    ):
        assert inactive not in result.stdout


@pytest.mark.parametrize("command", ACTIVE_COMMANDS)
def test_every_documented_command_has_real_help(command: str) -> None:
    result = _help(command)
    assert result.returncode == 0, result.stderr
    assert f"usage: so101-pusht-benchmark {command}" in result.stdout


def test_freeze_help_documents_artifact_free_metadata_planning() -> None:
    result = _help("freeze-experiment")
    assert result.returncode == 0
    assert "--metadata" in result.stdout
    assert "--dry-run" in result.stdout
    assert "does not probe the F710, create artifacts, or start training" in result.stdout


def test_readme_names_every_canonical_model_path_and_id() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    expected = (
        ("dp_cnn", "dp-cnn-production-smoke", "dp-cnn-production"),
        ("dp_transformer", "dp-transformer-production-smoke", "dp-transformer-production"),
        ("ibc", "ibc-production-smoke", "ibc-production"),
        ("lstm_gmm", "lstm-gmm-production-smoke", "lstm-gmm-production"),
    )
    for model, smoke_id, final_id in expected:
        assert smoke_id in readme
        assert final_id in readme
        assert f"models/{model}/full" in readme
        assert f"models/{model}/bundle/policy.safetensors" in readme
        assert f"evaluations/{model}/metrics.json" in readme
    assert "repeat similarly" not in readme.lower()
    assert "smoke IDs and smoke checkpoint paths never enter those commands" in " ".join(
        readme.split()
    )


def test_executable_runbook_dry_run_parses_and_never_routes_smoke_to_final(
    tmp_path: Path,
) -> None:
    script = PACKAGE_ROOT / "scripts/production_operator.sh"
    experiment = tmp_path / "user-selected-experiment.yaml"
    result = subprocess.run(
        [str(script), "--dry-run", str(experiment)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output_lines = result.stdout.splitlines()
    commands = [
        line.removeprefix("DRY_RUN: ") for line in output_lines if line.startswith("DRY_RUN: ")
    ]
    completion_notes = [line for line in output_lines if line.startswith("DRY_RUN_NOTE: ")]
    assert len(completion_notes) == 4
    assert commands
    assert all("$" not in command for command in commands)
    parser = command_parser()
    parsed: list[argparse.Namespace] = []
    for command in commands:
        words = shlex.split(command)
        marker = words.index("so101_pusht_benchmark.cli")
        parsed.append(parser.parse_args(words[marker + 1 :]))
    full = [item for item in parsed if item.command == "train-model" and item.full_production]
    smoke = [item for item in parsed if item.command == "train-model" and item.smoke_mode]
    exports = [item for item in parsed if item.command == "export-inference-bundle"]
    evaluations = [item for item in parsed if item.command == "evaluate-model"]
    compares = [item for item in parsed if item.command == "compare-models"]
    final_ids = [
        "dp-cnn-production",
        "dp-transformer-production",
        "ibc-production",
        "lstm-gmm-production",
    ]
    assert [item.artifact_id for item in smoke] == [f"{item}-smoke" for item in final_ids]
    assert [item.artifact_id for item in full if not item.preflight] == final_ids
    assert all(item.max_updates == 100_000 for item in full)
    assert [item.artifact_id for item in exports] == final_ids
    assert [item.artifact_id for item in evaluations] == final_ids
    assert len(compares) == 1
    assert compares[0].artifact_ids == final_ids
    assert all("/smoke/" not in str(item.checkpoint) for item in exports)


def test_four_model_route_uses_distinct_smoke_ids_and_stable_final_ids() -> None:
    parser = command_parser()
    models_and_ids = (
        ("dp_cnn", "dp-cnn-production"),
        ("dp_transformer", "dp-transformer-production"),
        ("ibc", "ibc-production"),
        ("lstm_gmm", "lstm-gmm-production"),
    )
    final_ids: list[str] = []
    for model, artifact_id in models_and_ids:
        common = [
            "--model",
            model,
            "--paper-view",
            "/data/frozen",
            "--artifact-index",
            "/artifacts/index.json",
        ]
        smoke = parser.parse_args(
            [
                "train-model",
                *common,
                "--output",
                f"/artifacts/smoke/{model}",
                "--artifact-id",
                f"{artifact_id}-smoke",
                "--smoke-mode",
                "production",
            ]
        )
        full = parser.parse_args(
            [
                "train-model",
                *common,
                "--output",
                f"/artifacts/models/{model}/full",
                "--artifact-id",
                artifact_id,
                "--full-production",
                "--max-updates",
                "100000",
            ]
        )
        export = parser.parse_args(
            [
                "export-inference-bundle",
                "--checkpoint",
                f"/artifacts/models/{model}/full/checkpoints/latest.ckpt",
                "--config",
                f"/artifacts/models/{model}/full/resolved_config.json",
                "--output",
                f"/artifacts/models/{model}/bundle",
                "--artifact-id",
                artifact_id,
                "--artifact-index",
                "/artifacts/index.json",
            ]
        )
        evaluate = parser.parse_args(
            [
                "evaluate-model",
                "--model",
                model,
                "--bundle",
                f"/artifacts/models/{model}/bundle/policy.safetensors",
                "--output",
                f"/artifacts/evaluations/{model}",
                "--artifact-id",
                artifact_id,
                "--artifact-index",
                "/artifacts/index.json",
            ]
        )
        assert smoke.artifact_id == f"{artifact_id}-smoke"
        assert smoke.smoke_mode == "production"
        assert full.artifact_id == export.artifact_id == evaluate.artifact_id == artifact_id
        assert full.full_production is True
        assert full.max_updates == 100_000
        final_ids.append(artifact_id)
    compare = parser.parse_args(
        [
            "compare-models",
            "--artifact-index",
            "/artifacts/index.json",
            *[value for artifact_id in final_ids for value in ("--artifact-id", artifact_id)],
            "--output",
            "/artifacts/reports/four-model-final",
        ]
    )
    assert compare.artifact_ids == final_ids
    assert all(f"{artifact_id}-smoke" not in compare.artifact_ids for artifact_id in final_ids)


def test_documented_operator_commands_parse_without_legacy_routes() -> None:
    parser = command_parser()
    examples = (
        ["inspect-env", "--native-pusht-so100"],
        ["collect-native", "--preflight", "--dataset-root", "/data/raw"],
        ["collect-native", "--launch", "--dataset-root", "/data/raw"],
        ["import-native", "--repo", "/data/raw", "--output", "/data/imported"],
        [
            "freeze-experiment",
            "--metadata",
            "/data/raw/meta/info.json",
            "--experiment-config",
            "/data/experiment.yaml",
            "--dry-run",
        ],
        [
            "freeze-experiment",
            "--source",
            "/data/imported",
            "--output",
            "/data/frozen",
            "--experiment-config",
            "/data/experiment.yaml",
        ],
        [
            "train-model",
            "--model",
            "dp_cnn",
            "--paper-view",
            "/data/frozen",
            "--output",
            "/artifacts/dp_cnn",
            "--artifact-id",
            "dp-cnn-production",
            "--artifact-index",
            "/artifacts/index.json",
            "--full-production",
            "--max-updates",
            "100000",
        ],
        [
            "export-inference-bundle",
            "--checkpoint",
            "/artifacts/dp_cnn/checkpoint.ckpt",
            "--config",
            "/artifacts/dp_cnn/resolved.json",
            "--output",
            "/artifacts/dp_cnn/bundle",
            "--artifact-id",
            "dp-cnn-production",
            "--artifact-index",
            "/artifacts/index.json",
        ],
        [
            "evaluate-model",
            "--model",
            "dp_cnn",
            "--bundle",
            "/artifacts/dp_cnn/bundle/policy.safetensors",
            "--output",
            "/evaluations/dp_cnn",
            "--artifact-id",
            "dp-cnn-production",
            "--artifact-index",
            "/artifacts/index.json",
        ],
        [
            "compare-models",
            "--artifact-index",
            "/artifacts/index.json",
            "--artifact-id",
            "dp-cnn-production",
            "--artifact-id",
            "dp-transformer-production",
            "--artifact-id",
            "ibc-production",
            "--artifact-id",
            "lstm-gmm-production",
            "--output",
            "/reports/four-model-final",
        ],
    )
    assert [parser.parse_args(example).command for example in examples] == [
        example[0] for example in examples
    ]
