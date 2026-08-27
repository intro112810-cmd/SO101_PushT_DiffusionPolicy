from __future__ import annotations

import builtins
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal, cast

import pytest
import yaml

from so101_pusht_benchmark import native_cli


PACKAGE = Path(__file__).parents[1]
HISTORICAL_CONFIG = PACKAGE / "configs/benchmark/pusht_v1.yaml"
SYSTEM_PYTHON = Path("/usr/bin/python3")
HISTORICAL_IMPORTS = (
    "so101_pusht_benchmark.collection.recorder",
    "so101_pusht_benchmark.collection.viewer",
    "so101_pusht_benchmark.data.exporter",
    "so101_pusht_benchmark.sim.env",
    "so101_pusht_benchmark.task.spec",
)


def run_module(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(PACKAGE / "src")}
    return subprocess.run(
        [sys.executable, "-m", "so101_pusht_benchmark.cli", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_main_help_exposes_only_native_active_commands() -> None:
    result = run_module("--help", cwd=PACKAGE)

    assert result.returncode == 0
    for command in (
        "validate-contract",
        "inspect-env",
        "collect-native",
        "export-native",
        "train-model",
        "compare-models",
    ):
        assert command in result.stdout
    for historical in (
        "validate-sim",
        "step-smoke",
        "calibrate-sim",
        "collect-sim",
        "export-paper-view",
    ):
        assert historical not in result.stdout


def test_train_model_help_and_mode_selection_are_unambiguous() -> None:
    result = run_module("train-model", "--help", cwd=PACKAGE)
    assert result.returncode == 0
    assert "--smoke-mode {fixture,production}" in result.stdout
    assert "--full-production" in result.stdout
    assert "--max-updates" in result.stdout
    assert "100000-update bound" in " ".join(result.stdout.split())
    assert "200-step" not in result.stdout

    parser = native_cli.command_parser()
    common = [
        "train-model",
        "--model",
        "dp_cnn",
        "--paper-view",
        "/store",
        "--output",
        "/output",
        "--artifact-id",
        "id",
        "--artifact-index",
        "/index.json",
    ]
    fixture = parser.parse_args([*common, "--smoke"])
    production = parser.parse_args([*common, "--smoke-mode", "production"])
    full = parser.parse_args([*common, "--full-production", "--max-updates", "100000"])
    assert fixture.smoke is True
    assert fixture.smoke_mode is None
    assert production.smoke is False
    assert production.smoke_mode == "production"
    assert full.full_production is True
    assert full.max_updates == 100_000
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--smoke", "--smoke-mode", "production"])


def test_missing_native_runtime_fails_closed_before_historical_imports(tmp_path: Path) -> None:
    dataset_root = (
        PACKAGE.parents[1]
        / "04_experiments/so101_pusht_benchmark/datasets"
        / f".pytest-{tmp_path.name}-must-not-exist"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SO101_PUSHT_F710_DEVICES_JSON": "[]",
    }

    result = subprocess.run(
        [
            str(SYSTEM_PYTHON),
            "-m",
            "so101_pusht_benchmark.cli",
            "collect-native",
            "--preflight",
            "--dataset-root",
            str(dataset_root),
        ],
        cwd=PACKAGE,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout.startswith("FAIL CLOSED: native pushT-so100 runtime mismatch")
    assert "Traceback" not in result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stdout + result.stderr
    assert not dataset_root.exists()


def test_real_module_native_preflight_does_not_load_historical_modules(tmp_path: Path) -> None:
    dataset_root = (
        PACKAGE.parents[1]
        / "04_experiments/so101_pusht_benchmark/datasets"
        / f".pytest-{tmp_path.name}-must-not-exist"
    )
    script = (
        "import json,runpy,sys;"
        f"sys.argv=['so101-pusht-benchmark','collect-native','--preflight','--dataset-root',{str(dataset_root)!r}];"
        "code=0;"
        "\ntry: runpy.run_module('so101_pusht_benchmark.cli',run_name='__main__')"
        "\nexcept SystemExit as exc: code=exc.code;"
        f"\nprint('MODULE_EXIT='+str(code));print('HISTORICAL_LOADED='+json.dumps([name for name in {HISTORICAL_IMPORTS!r} if name in sys.modules]))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SO101_PUSHT_F710_DEVICES_JSON": (
            '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11}]'
        ),
        "SO101_PUSHT_DISPLAY_AVAILABLE": "1",
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert '"status": "ready"' in result.stdout
    assert "MODULE_EXIT=0" in result.stdout
    assert "HISTORICAL_LOADED=[]" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
    assert not dataset_root.exists()


def test_validate_native_contract_from_repository_and_package_cwds() -> None:
    for cwd in (PACKAGE.parents[1], PACKAGE):
        result = run_module("validate-contract", cwd=cwd)
        assert result.returncode == 0
        assert "contract.identifier=pusht_so100_native_v1" in result.stdout
        assert "contract.observation=cam_top:uint8[3,224,224]" in result.stdout
        assert "agent_pos:float32[5]" in result.stdout
        assert "contract.action=absolute_mocap_xy:float32[2]" in result.stdout


def test_historical_explicit_config_and_removed_command_fail_inactive() -> None:
    historical = run_module(
        "validate-contract",
        "--config",
        str(HISTORICAL_CONFIG),
        cwd=PACKAGE,
    )
    removed = run_module(
        "collect-sim",
        "--synthetic-pipeline-probe",
        "--ticks",
        "1",
        "--max-attempts",
        "1",
        cwd=PACKAGE,
    )

    assert historical.returncode == 1
    assert "historical/inactive config" in historical.stdout
    assert "state:float32[15]" not in historical.stdout
    assert removed.returncode == 2
    assert "invalid choice" in removed.stderr


def test_native_collection_and_export_preflights_never_import_historical_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report: native_cli.RuntimeReport = {
        "status": "compatible",
        "plan": ".omo/plans/pusht-so100-four-model-clean-restart.md",
        "contract_schema": "pusht-so100-native-v1",
        "lock": "environments/sim-runtime.lock",
        "lock_sha256": "a" * 64,
        "source_environment_sha256": "b" * 64,
        "fallback": "forbidden",
        "runtime": {},
        "upstream": {
            "head": "f4d6d1311bc0b43ce65458a9edd856f3c7e0a520",
            "remote": "https://github.com/boaoqian/pushT-so100.git",
            "environment_sha256": "b" * 64,
            "runtime_manifest_sha256": "c" * 64,
            "runtime_member_count": 23,
            "approved_patches": ["src/env_human_ee.py"],
            "excluded_untracked": ["MUJOCO_LOG.TXT"],
        },
    }
    monkeypatch.setattr(native_cli, "native_runtime_report", lambda: report)
    monkeypatch.setenv(
        "SO101_PUSHT_F710_DEVICES_JSON",
        '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11}]',
    )
    monkeypatch.setenv("SO101_PUSHT_DISPLAY_AVAILABLE", "1")
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        forbidden = (
            "so101_pusht_benchmark.task.spec",
            "so101_pusht_benchmark.sim.env",
            "so101_pusht_benchmark.data.exporter",
        )
        if name in forbidden:
            raise AssertionError(f"historical pipeline import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    collection_buffer = StringIO()
    with redirect_stdout(collection_buffer):
        assert native_cli.main(["collect-native", "--preflight"]) == 0
    collection = collection_buffer.getvalue()
    assert '"command": "collect-native"' in collection
    assert '"adapter": "frozen_pushT_so100"' in collection
    export_buffer = StringIO()
    with redirect_stdout(export_buffer):
        assert native_cli.main(["export-native", "--preflight"]) == 0
    export = export_buffer.getvalue()
    assert '"command": "export-native"' in export
    assert "selected_view" not in export
    assert "zarr" not in export


def _copied_active_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    relative = {
        "benchmark": Path("configs/benchmark/pusht_so100_native_v1.yaml"),
        "collection": Path("configs/collection/pusht_so100_f710_native_v1.yaml"),
        "export": Path("configs/export/pusht_so100_native_v1.yaml"),
    }
    copied: dict[str, Path] = {}
    for role, path in relative.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PACKAGE / path).read_bytes())
        copied[role] = destination
    monkeypatch.setattr(native_cli, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(native_cli, "_BENCHMARK_CONFIG", copied["benchmark"])
    monkeypatch.setattr(native_cli, "_COLLECTION_CONFIG", copied["collection"])
    monkeypatch.setattr(native_cli, "_EXPORT_CONFIG", copied["export"])
    return copied


def _mutate_yaml(path: Path, mutation: tuple[str, ...], value: object) -> None:
    loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = cast("dict[str, object]", loaded)
    target = raw
    for key in mutation[:-1]:
        target = cast("dict[str, object]", target[key])
    leaf = mutation[-1]
    if value is _MISSING:
        del target[leaf]
    else:
        target[leaf] = value
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


_MISSING = object()
ConfigRole = Literal["benchmark", "collection", "export"]


@dataclass(frozen=True)
class MutationCase:
    role: ConfigRole
    mutation: tuple[str, ...]
    value: object
    argv: tuple[str, ...]


_MUTATION_CASES: list[MutationCase] = [
    MutationCase("benchmark", ("observation", "agent_pos", "shape"), [15], ("validate-contract",)),
    MutationCase(
        "collection",
        ("controller", "axes", "rotation"),
        2,
        ("collect-native", "--preflight"),
    ),
    MutationCase("export", ("keys",), ["selected_view"], ("export-native", "--preflight")),
    MutationCase("benchmark", ("horizon",), _MISSING, ("validate-contract",)),
    MutationCase("benchmark", ("unknown",), True, ("validate-contract",)),
    MutationCase(
        "collection",
        ("controller", "button_debounce_seconds"),
        _MISSING,
        ("collect-native", "--preflight"),
    ),
    MutationCase(
        "collection",
        ("controller", "legacy_axis"),
        2,
        ("collect-native", "--preflight"),
    ),
    MutationCase("export", ("transforms",), _MISSING, ("export-native", "--preflight")),
    MutationCase("export", ("selected_view",), "top", ("export-native", "--preflight")),
    MutationCase("export", ("zarr",), {}, ("export-native", "--preflight")),
]


@pytest.mark.parametrize("case", _MUTATION_CASES)
def test_exact_active_path_mutations_fail_without_compatible_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: MutationCase,
) -> None:
    copied = _copied_active_configs(tmp_path, monkeypatch)
    _mutate_yaml(copied[case.role], case.mutation, case.value)
    monkeypatch.setattr(
        native_cli,
        "native_runtime_report",
        lambda: (_ for _ in ()).throw(AssertionError("runtime reached before config acceptance")),
    )
    output = StringIO()

    with redirect_stdout(output):
        result = native_cli.main(list(case.argv))

    assert result == 1
    assert output.getvalue().startswith("FAIL CLOSED:")
    assert "compatible" not in output.getvalue()


def test_malformed_config_fails_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("schema: 1\n", encoding="utf-8")
    failed = run_module("validate-contract", "--config", str(malformed), cwd=PACKAGE)

    assert failed.returncode == 1
    assert failed.stdout.startswith("FAIL CLOSED:")
