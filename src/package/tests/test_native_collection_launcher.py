from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from so101_pusht_benchmark import native_cli
from so101_pusht_benchmark.collection import native as native_collection
from so101_pusht_benchmark.collection.native import (
    DEVICE_PROBE_ENV,
    DISPLAY_PROBE_ENV,
    F710Device,
    NativeCollectionError,
    NativeCollectionPlan,
    NativeCollectionRequest,
    launch_native_collection,
    preflight_native_collection,
)
from so101_pusht_benchmark.native_runtime import NativeRuntimeError


PACKAGE_ROOT = Path(__file__).parents[1]
UPSTREAM_ROOT = PACKAGE_ROOT.parents[1] / "05_references/external_repos/pushT-so100"
RUNTIME_REPORT = {
    "status": "compatible",
    "plan": ".omo/plans/pusht-so100-four-model-clean-restart.md",
    "contract_schema": "pusht-so100-native-v1",
    "lock": "environments/sim-runtime.lock",
    "lock_sha256": "a" * 64,
    "source_environment_sha256": "b" * 64,
    "fallback": "forbidden",
    "runtime": {"python": "3.10", "pygame": "2.6.1"},
}
F710 = F710Device(name="Logitech Gamepad F710", axes=6, buttons=11)


def _request(dataset_root: Path) -> NativeCollectionRequest:
    return NativeCollectionRequest(
        dataset_root=dataset_root,
        fps=10,
        move_speed=0.05,
        rotation_speed=1.0,
    )


def _preflight(
    dataset_root: Path,
    *,
    devices: tuple[F710Device, ...] = (F710,),
    display: bool = True,
    environment: dict[str, str] | None = None,
) -> NativeCollectionPlan:
    selected_environment = dict(
        environment or {"DISPLAY": ":77", "PATH": "/usr/bin", "KEEP": "yes"}
    )
    selected_environment[DISPLAY_PROBE_ENV] = "1" if display else "0"
    return preflight_native_collection(
        _request(dataset_root),
        runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
        device_probe=lambda: devices,
        environment=selected_environment,
        executable="/native/bin/python",
        allowed_dataset_root=dataset_root.parent,
    )


def test_preflight_produces_exact_frozen_cwd_argv_and_environment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    environment = {
        "DISPLAY": ":77",
        "PATH": "/usr/bin",
        "KEEP": "yes",
        DEVICE_PROBE_ENV: '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11}]',
    }

    plan = _preflight(dataset_root, environment=environment)

    assert plan.cwd == UPSTREAM_ROOT
    assert plan.argv == (
        "/native/bin/python",
        "src/env_human_ee.py",
        "--repo_id",
        str(dataset_root),
        "--fps",
        "10",
        "--move_speed",
        "0.05",
        "--rot_speed",
        "1.0",
    )
    assert plan.environment == {
        "DISPLAY": ":77",
        "PATH": "/home/intro/miniforge3/envs/so100test/bin:/usr/bin:/bin",
        "MUJOCO_GL": "egl",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    assert plan.device == F710
    assert plan.runtime == RUNTIME_REPORT
    assert not dataset_root.exists()


def test_preflight_is_idempotent_and_preserves_existing_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    dataset_root.mkdir()
    marker = dataset_root / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    first = _preflight(dataset_root)
    second = _preflight(dataset_root)

    assert first == second
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ((), "F710 joystick unavailable"),
        ((F710Device(name="Xbox Controller", axes=6, buttons=11),), "F710 joystick unavailable"),
        (
            (F710Device(name="Logitech Gamepad F710", axes=3, buttons=11),),
            "F710 joystick unavailable",
        ),
        (
            (F710Device(name="Logitech Gamepad F710", axes=6, buttons=7),),
            "F710 joystick unavailable",
        ),
    ],
)
def test_preflight_fails_closed_for_no_exact_capable_f710_without_output(
    tmp_path: Path,
    devices: tuple[F710Device, ...],
    message: str,
) -> None:
    dataset_root = tmp_path / "episodes"

    with pytest.raises(NativeCollectionError, match=message):
        _preflight(dataset_root, devices=devices)

    assert not dataset_root.exists()


def test_preflight_rejects_capable_f710_when_recorder_index_zero_is_different(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "episodes"
    devices = (
        F710Device(name="Xbox Controller", axes=6, buttons=11),
        F710,
    )

    with pytest.raises(
        NativeCollectionError,
        match="joystick index 0 must be a capable Logitech Gamepad F710",
    ):
        _preflight(dataset_root, devices=devices)

    assert not dataset_root.exists()


def test_preflight_fails_closed_for_missing_display_without_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"

    with pytest.raises(NativeCollectionError, match="graphical display unavailable"):
        _preflight(dataset_root, display=False)

    assert not dataset_root.exists()


@pytest.mark.parametrize(
    ("request_update", "message"),
    [
        ({"fps": 9}, "FPS must be exactly 10"),
        ({"fps": True}, "FPS must be exactly 10"),
        ({"move_speed": 0.0}, "move speed must be exactly 0.05"),
        ({"move_speed": float("nan")}, "move speed must be exactly 0.05"),
        ({"rotation_speed": 2.0}, "rotation speed must be exactly 1.0"),
        ({"rotation_speed": float("inf")}, "rotation speed must be exactly 1.0"),
    ],
)
def test_preflight_rejects_malformed_timing_or_speeds_before_probe(
    tmp_path: Path,
    request_update: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "dataset_root": tmp_path / "episodes",
        "fps": 10,
        "move_speed": 0.05,
        "rotation_speed": 1.0,
    }
    values.update(request_update)
    request = NativeCollectionRequest(
        dataset_root=cast("Path", values["dataset_root"]),
        fps=cast("int", values["fps"]),
        move_speed=cast("float", values["move_speed"]),
        rotation_speed=cast("float", values["rotation_speed"]),
    )
    probe_called = False

    def device_probe() -> tuple[F710Device, ...]:
        nonlocal probe_called
        probe_called = True
        return (F710,)

    with pytest.raises(NativeCollectionError, match=message):
        preflight_native_collection(
            request,
            runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
            device_probe=device_probe,
            environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
            executable="/native/bin/python",
            allowed_dataset_root=tmp_path,
        )

    assert probe_called is False
    assert not cast("Path", values["dataset_root"]).exists()


def test_preflight_rejects_unusable_dataset_parent_before_device_probe(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing-parent" / "episodes"
    probe_called = False

    def device_probe() -> tuple[F710Device, ...]:
        nonlocal probe_called
        probe_called = True
        return (F710,)

    with pytest.raises(NativeCollectionError, match="dataset parent unavailable"):
        preflight_native_collection(
            _request(dataset_root),
            runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
            device_probe=device_probe,
            environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
            executable="/native/bin/python",
            allowed_dataset_root=tmp_path,
        )

    assert probe_called is False
    assert not dataset_root.exists()


def test_runtime_failure_prevents_device_probe_and_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    probe_called = False

    def failed_runtime() -> dict[str, object]:
        raise NativeRuntimeError("Python: expected 3.10, found 3.12")

    def device_probe() -> tuple[F710Device, ...]:
        nonlocal probe_called
        probe_called = True
        return (F710,)

    with pytest.raises(NativeRuntimeError, match=r"expected 3\.10"):
        preflight_native_collection(
            _request(dataset_root),
            runtime_report=failed_runtime,
            device_probe=device_probe,
            environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
            allowed_dataset_root=tmp_path,
        )

    assert probe_called is False
    assert not dataset_root.exists()


def test_existing_file_is_rejected_without_mutation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    dataset_root.write_text("not a dataset", encoding="utf-8")

    with pytest.raises(NativeCollectionError, match="dataset path is not a directory"):
        _preflight(dataset_root)

    assert dataset_root.read_text(encoding="utf-8") == "not a dataset"


def test_launcher_passes_exact_spec_and_propagates_partial_failure(tmp_path: Path) -> None:
    plan = _preflight(tmp_path / "episodes")
    received: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def runner(argv: tuple[str, ...], cwd: Path, environment: dict[str, str]) -> int:
        received.append((argv, cwd, environment))
        return 23

    assert launch_native_collection(plan, runner=runner, provenance_verifier=dict) == 23
    assert received == [(plan.argv, plan.cwd, plan.environment)]
    assert not (tmp_path / "episodes").exists()


def test_cli_happy_preflight_prints_exact_command_without_creating_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    dataset_root = datasets / "episodes"
    monkeypatch.setattr(native_collection, "runtime_artifact_root", lambda: tmp_path)
    monkeypatch.setenv(
        DEVICE_PROBE_ENV,
        '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11}]',
    )
    monkeypatch.setenv(DISPLAY_PROBE_ENV, "1")
    output = StringIO()

    with redirect_stdout(output):
        result = native_cli.main(
            ["collect-native", "--preflight", "--dataset-root", str(dataset_root)]
        )

    report = json.loads(output.getvalue())
    assert result == 0
    assert report["status"] == "ready"
    assert report["cwd"] == str(UPSTREAM_ROOT)
    assert report["argv"] == [
        sys.executable,
        "src/env_human_ee.py",
        "--repo_id",
        str(dataset_root),
        "--fps",
        "10",
        "--move_speed",
        "0.05",
        "--rot_speed",
        "1.0",
    ]
    assert report["mapping"]["axes"] == {"x": 0, "y": 1, "rotation": 3}
    assert report["mapping"]["buttons"] == {
        "z_up": 4,
        "z_down": 0,
        "reset": 3,
        "record_toggle": 1,
        "exit": 7,
    }
    assert not dataset_root.exists()


def test_real_probe_preflight_suppresses_only_pygame_pkg_resources_warning(
    tmp_path: Path,
) -> None:
    fake_packages = tmp_path / "fake-packages"
    pygame = fake_packages / "pygame"
    pygame.mkdir(parents=True)
    pygame.joinpath("__init__.py").write_text(
        "from . import joystick, pkgdata\n",
        encoding="utf-8",
    )
    pygame.joinpath("pkgdata.py").write_text(
        "import warnings\nwarnings.warn(\n    'pkg_resources is deprecated as an API. See '",
        encoding="utf-8",
    )
    with pygame.joinpath("pkgdata.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "'https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources '",
        )
        stream.write(
            "'package is slated for removal as early as 2025-11-30. Refrain from using '",
        )
        stream.write("'this package or pin to Setuptools<81.', UserWarning)\n")
    pygame.joinpath("joystick.py").write_text(
        "import warnings\n"
        "warnings.warn('real joystick diagnostic', RuntimeWarning)\n"
        "def init(): pass\n"
        "def quit(): pass\n"
        "def get_count(): return 1\n"
        "class Joystick:\n"
        "    def __init__(self, index): assert index == 0\n"
        "    def get_name(self): return 'Logitech Gamepad F710'\n"
        "    def get_numaxes(self): return 6\n"
        "    def get_numbuttons(self): return 11\n",
        encoding="utf-8",
    )
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    dataset_root = datasets / "episodes"
    script = (
        "import json;"
        "from pathlib import Path;"
        "from so101_pusht_benchmark import native_cli;"
        "from so101_pusht_benchmark.collection import native;"
        f"native.runtime_artifact_root=lambda:Path({str(tmp_path)!r});"
        f"native_cli.native_runtime_report=lambda:{RUNTIME_REPORT!r};"
        f"raise SystemExit(native_cli.main(['collect-native','--preflight','--dataset-root',{str(dataset_root)!r}]))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(fake_packages), str(PACKAGE_ROOT / "src"))),
        DISPLAY_PROBE_ENV: "1",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.pop(DEVICE_PROBE_ENV, None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["status"] == "ready"
    assert report["device"] == {"name": "Logitech Gamepad F710", "axes": 6, "buttons": 11}
    assert "pkg_resources is deprecated as an API" not in result.stderr
    assert "RuntimeWarning: real joystick diagnostic" in result.stderr
    assert not dataset_root.exists()


def test_cli_no_device_fails_nonzero_without_creating_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    dataset_root = datasets / "episodes"
    monkeypatch.setattr(native_collection, "runtime_artifact_root", lambda: tmp_path)
    monkeypatch.setenv(DEVICE_PROBE_ENV, "[]")
    monkeypatch.setenv(DISPLAY_PROBE_ENV, "1")
    output = StringIO()

    with redirect_stdout(output):
        result = native_cli.main(
            ["collect-native", "--preflight", "--dataset-root", str(dataset_root)]
        )

    assert result != 0
    assert output.getvalue() == "FAIL CLOSED: F710 joystick unavailable\n"
    assert not dataset_root.exists()


def test_cli_injected_probe_cannot_cross_the_launch_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    dataset_root = datasets / "episodes"
    monkeypatch.setattr(native_collection, "runtime_artifact_root", lambda: tmp_path)
    monkeypatch.setenv(
        DEVICE_PROBE_ENV,
        '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11}]',
    )
    monkeypatch.setenv(DISPLAY_PROBE_ENV, "1")
    output = StringIO()

    with redirect_stdout(output):
        result = native_cli.main(
            ["collect-native", "--launch", "--dataset-root", str(dataset_root)]
        )

    assert result == 1
    assert output.getvalue() == "FAIL CLOSED: injected probes are preflight-only\n"
    assert not dataset_root.exists()


def test_preflight_rejects_symlink_dataset_and_ancestor_before_other_checks(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    final_link = allowed / "final-link"
    final_link.symlink_to(outside, target_is_directory=True)
    ancestor_link = allowed / "ancestor-link"
    ancestor_link.symlink_to(outside, target_is_directory=True)
    runtime_calls = 0
    probe_calls = 0

    def runtime_report() -> dict[str, object]:
        nonlocal runtime_calls
        runtime_calls += 1
        return cast("dict[str, object]", RUNTIME_REPORT)

    def device_probe() -> tuple[F710Device, ...]:
        nonlocal probe_calls
        probe_calls += 1
        return (F710,)

    for candidate in (final_link, ancestor_link / "episodes"):
        with pytest.raises(NativeCollectionError, match="symlink"):
            preflight_native_collection(
                _request(candidate),
                runtime_report=runtime_report,
                device_probe=device_probe,
                environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
                allowed_dataset_root=allowed,
            )

    assert runtime_calls == probe_calls == 0
    assert not (outside / "episodes").exists()


@pytest.mark.parametrize(
    "relative",
    ["data", "images", "meta", "videos", "images/chunk-000"],
)
def test_preflight_rejects_symlinked_existing_dataset_descendants(
    tmp_path: Path,
    relative: str,
) -> None:
    dataset_root = tmp_path / "episodes"
    dataset_root.mkdir()
    outside = tmp_path / "outside-descendant"
    outside.mkdir()
    link = dataset_root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(NativeCollectionError, match="dataset tree contains symlink"):
        _preflight(dataset_root)


def test_preflight_rejects_traversal_and_paths_outside_canonical_dataset_root(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    (allowed / "nested").mkdir()

    for candidate in (allowed / "nested" / ".." / "episodes", tmp_path / "outside"):
        with pytest.raises(NativeCollectionError, match="dataset root"):
            preflight_native_collection(
                _request(candidate),
                runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
                device_probe=lambda: (F710,),
                environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
                allowed_dataset_root=allowed,
            )


def test_launch_revalidates_path_before_provenance_or_subprocess(tmp_path: Path) -> None:
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    dataset_root = allowed / "episodes"
    plan = preflight_native_collection(
        _request(dataset_root),
        runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
        device_probe=lambda: (F710,),
        environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
        executable="/native/bin/python",
        allowed_dataset_root=allowed,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    dataset_root.symlink_to(outside, target_is_directory=True)
    provenance_calls = 0
    process_calls = 0

    def verify() -> dict[str, object]:
        nonlocal provenance_calls
        provenance_calls += 1
        return {}

    def runner(argv: tuple[str, ...], cwd: Path, environment: dict[str, str]) -> int:
        del argv, cwd, environment
        nonlocal process_calls
        process_calls += 1
        return 0

    with pytest.raises(NativeCollectionError, match="symlink"):
        launch_native_collection(plan, runner=runner, provenance_verifier=verify)

    assert provenance_calls == process_calls == 0
    assert list(outside.iterdir()) == []


def test_upstream_mutation_fails_immediately_before_process_without_output(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "datasets"
    allowed.mkdir()
    dataset_root = allowed / "episodes"
    plan = preflight_native_collection(
        _request(dataset_root),
        runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
        device_probe=lambda: (F710,),
        environment={"DISPLAY": ":77", DISPLAY_PROBE_ENV: "1"},
        executable="/native/bin/python",
        allowed_dataset_root=allowed,
    )
    upstream_member = tmp_path / "env_human_ee.py"
    upstream_member.write_text("approved", encoding="utf-8")
    approved = upstream_member.read_bytes()
    upstream_member.write_text("mutated", encoding="utf-8")
    events: list[str] = []

    def verify() -> dict[str, object]:
        events.append("provenance")
        if upstream_member.read_bytes() != approved:
            raise NativeCollectionError("undeclared upstream drift")
        return {}

    def runner(argv: tuple[str, ...], cwd: Path, environment: dict[str, str]) -> int:
        del argv, cwd, environment
        events.append("process")
        return 0

    with pytest.raises(NativeCollectionError, match="upstream drift"):
        launch_native_collection(plan, runner=runner, provenance_verifier=verify)

    assert events == ["provenance"]
    assert not dataset_root.exists()


def test_child_environment_drops_secrets_tokens_and_python_injection(tmp_path: Path) -> None:
    sentinels = {
        "AWS_SECRET_ACCESS_KEY": "must-not-leak",
        "GITHUB_TOKEN": "must-not-leak",
        "HF_TOKEN": "must-not-leak",
        "PYTHONPATH": "/attacker",
        "PYTHONHOME": "/attacker",
        "PYTHONINSPECT": "1",
        "LD_PRELOAD": "/attacker.so",
        "ARBITRARY_OPERATOR_VALUE": "must-not-leak",
    }
    plan = _preflight(
        tmp_path / "episodes",
        environment={
            **sentinels,
            "DISPLAY": ":77",
            "XAUTHORITY": "/run/user/1000/xauth",
            "PATH": "/native/bin:/usr/bin",
            "LANG": "C.UTF-8",
        },
    )

    assert not set(sentinels) & set(plan.environment)
    assert plan.environment["DISPLAY"] == ":77"
    assert plan.environment["XAUTHORITY"] == "/run/user/1000/xauth"
    assert plan.environment["PATH"] == (
        "/home/intro/miniforge3/envs/so100test/bin:/usr/bin:/bin"
    )
    assert plan.environment["LANG"] == "C.UTF-8"


def test_injected_device_json_is_strict_and_never_reaches_child_environment(tmp_path: Path) -> None:
    environment = {
        "DISPLAY": ":77",
        DEVICE_PROBE_ENV: '[{"name":"Logitech Gamepad F710","axes":6,"buttons":11,"profile":"fallback"}]',
    }
    with pytest.raises(NativeCollectionError, match="malformed injected joystick probe"):
        preflight_native_collection(
            _request(tmp_path / "episodes"),
            runtime_report=lambda: cast("dict[str, object]", RUNTIME_REPORT),
            environment={**environment, DISPLAY_PROBE_ENV: "1"},
            executable="/native/bin/python",
            allowed_dataset_root=tmp_path,
        )
