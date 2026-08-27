from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from so101_pusht_benchmark.core.upstream_provenance import (
    UpstreamProvenanceError,
    UpstreamProvenanceReport,
)
from so101_pusht_benchmark.native_runtime import (
    NativeRuntimeError,
    assert_native_runtime,
    load_native_runtime_lock,
    native_runtime_report,
)
from so101_pusht_benchmark.workspace import PACKAGE_ROOT, load_workspace_policy


EXPECTED = {
    "python": "3.10",
    "lerobot": "0.4.4",
    "feetech-servo-sdk": "1.0.0",
    "gymnasium": "1.2.2",
    "mujoco": "3.3.7",
    "pygame": "2.6.1",
    "opencv-python": "5.0.0.93",
    "opencv-python-headless": "4.12.0.88",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "scipy": "1.15.3",
    "imageio": "2.37.4",
    "imageio-ffmpeg": "0.6.0",
    "av": "15.1.0",
    "pillow": "12.3.0",
}


def test_installed_cv2_resolves_to_declared_gui_build() -> None:
    cv2 = importlib.import_module("cv2")
    assert isinstance(cv2, ModuleType)
    assert importlib.metadata.version("opencv-python") == EXPECTED["opencv-python"]
    assert EXPECTED["opencv-python"].startswith(f"{cv2.__version__}.")
    gui_lines = [
        line.strip()
        for line in cv2.getBuildInformation().splitlines()
        if line.strip().startswith("GUI:")
    ]
    assert gui_lines
    assert gui_lines[0] != "GUI:                           NONE"
    assert callable(cv2.imshow)


def test_native_lock_matches_frozen_environment_and_policy() -> None:
    policy = load_workspace_policy()
    lock_path = PACKAGE_ROOT / policy["runtime"]["native_lock"]
    lock = load_native_runtime_lock(lock_path)
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    sidecar = (lock_path.parent / "sim-runtime.lock.sha256").read_text().split()[0]

    assert lock["contract_schema"] == "pusht-so100-native-v1"
    assert lock["required"] == EXPECTED
    assert lock["source_environment"] == {
        "path": "05_references/external_repos/pushT-so100/environment.yml",
        "sha256": "a7aab5a14bb18b6bb94cd1ecf13616384c6af87ba131ae5dc86fec7e94920f70",
        "preservation": "frozen_unchanged",
    }
    assert digest == sidecar == policy["runtime"]["native_lock_sha256"]


def test_native_preflight_accepts_exact_runtime_and_reports_governance() -> None:
    lock = load_native_runtime_lock()
    assert_native_runtime(actual=EXPECTED, lock=lock)
    report = native_runtime_report(actual=EXPECTED, lock=lock)

    assert report["status"] == "compatible"
    assert report["fallback"] == "forbidden"
    assert report["plan"] == ".omo/plans/pusht-so100-four-model-clean-restart.md"
    assert report["contract_schema"] == "pusht-so100-native-v1"
    assert report["lock"] == "environments/sim-runtime.lock"
    assert len(report["lock_sha256"]) == 64
    assert report["source_environment_sha256"] == (
        "a7aab5a14bb18b6bb94cd1ecf13616384c6af87ba131ae5dc86fec7e94920f70"
    )
    assert report["upstream"]["head"] == "f4d6d1311bc0b43ce65458a9edd856f3c7e0a520"
    assert report["upstream"]["runtime_member_count"] == 23


def test_native_runtime_report_classifies_strict_upstream_mismatch() -> None:
    calls = 0

    def failed_upstream(manifest: str | Path, root: str | Path) -> UpstreamProvenanceReport:
        del manifest, root
        nonlocal calls
        calls += 1
        raise UpstreamProvenanceError("undeclared upstream drift: mutated entrypoint")

    with pytest.raises(NativeRuntimeError, match="mutated entrypoint"):
        native_runtime_report(
            actual=EXPECTED,
            lock=load_native_runtime_lock(),
            upstream_verifier=failed_upstream,
        )

    assert calls == 1


def test_native_preflight_rejects_old_collection_runtime_without_fallback() -> None:
    old = dict(EXPECTED)
    old.update({"python": "3.12", "lerobot": "0.6.1", "gymnasium": "1.3.0", "mujoco": "3.8.1"})

    message = ""
    try:
        assert_native_runtime(actual=old)
    except NativeRuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("old collection runtime was accepted")
    assert "Python: expected 3.10, found 3.12" in message
    assert "LeRobot: expected 0.4.4, found 0.6.1" in message
    assert "Gymnasium: expected 1.2.2, found 1.3.0" in message
    assert "MuJoCo: expected 3.3.7, found 3.8.1" in message
    assert "fallback forbidden" in message
    assert "compatible" not in message


def test_native_lock_rejects_malformed_or_stale_state(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.lock"
    malformed.write_text("required: []\n", encoding="utf-8")
    with pytest.raises(NativeRuntimeError, match="malformed native runtime lock"):
        load_native_runtime_lock(malformed)

    raw = yaml.safe_load((PACKAGE_ROOT / "environments/sim-runtime.lock").read_text())
    raw["required"]["mujoco"] = "3.8.1"
    stale = tmp_path / "stale.lock"
    stale.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(NativeRuntimeError, match=r"MuJoCo pin must be 3\.3\.7"):
        load_native_runtime_lock(stale)
