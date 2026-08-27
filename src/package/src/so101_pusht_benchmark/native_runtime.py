"""Fail-closed runtime lock for native pushT-so100 collection and evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypedDict, cast

import yaml

from .core.upstream_provenance import (
    UpstreamProvenanceError,
    UpstreamProvenanceReport,
    verify_pusht_so100,
)


PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_LOCK = PACKAGE_ROOT / "environments/sim-runtime.lock"
UPSTREAM_MANIFEST = PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json"
UPSTREAM_ROOT = PACKAGE_ROOT.parents[1] / "05_references/external_repos/pushT-so100"
PLAN = ".omo/plans/pusht-so100-four-model-clean-restart.md"
CONTRACT_SCHEMA = "pusht-so100-native-v1"
_LOCK_RELATIVE = "environments/sim-runtime.lock"
_EXPECTED = {
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
_DISPLAY_NAMES = {
    "python": "Python",
    "lerobot": "LeRobot",
    "feetech-servo-sdk": "Feetech Servo SDK",
    "gymnasium": "Gymnasium",
    "mujoco": "MuJoCo",
    "pygame": "pygame",
    "opencv-python": "OpenCV GUI (opencv-python)",
    "opencv-python-headless": "OpenCV (opencv-python-headless)",
    "torch": "PyTorch",
    "torchvision": "Torchvision",
    "scipy": "SciPy",
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio-ffmpeg",
    "av": "PyAV",
    "pillow": "Pillow",
}


class NativeRuntimeError(RuntimeError):
    """Raised when the native runtime lock or current runtime is incompatible."""


class SourceEnvironment(TypedDict):
    path: str
    sha256: str
    preservation: str


class NativeRuntimeLock(TypedDict):
    schema: int
    lock_type: str
    role: list[str]
    platform: str
    contract_schema: str
    fallback: str
    required: dict[str, str]
    source_environment: SourceEnvironment


UpstreamVerifier = Callable[[str | Path, str | Path], UpstreamProvenanceReport]


class NativeRuntimeReport(TypedDict):
    status: str
    plan: str
    contract_schema: str
    lock: str
    lock_sha256: str
    source_environment_sha256: str
    fallback: str
    runtime: dict[str, str]
    upstream: UpstreamProvenanceReport


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NativeRuntimeError("malformed native runtime lock: root must be a mapping")
    return cast("dict[str, object]", value)


def load_native_runtime_lock(path: Path = DEFAULT_LOCK) -> NativeRuntimeLock:
    """Load the dedicated native lock and reject stale or malformed contracts."""
    try:
        raw_value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise NativeRuntimeError(f"malformed native runtime lock: {path}") from exc
    raw = _mapping(raw_value)
    required = raw.get("required")
    source = raw.get("source_environment")
    if (
        raw.get("schema") != 2
        or raw.get("lock_type") != "native_collection_evaluation"
        or raw.get("role") != ["collection", "evaluation"]
        or raw.get("platform") != "linux-64"
        or raw.get("contract_schema") != CONTRACT_SCHEMA
        or raw.get("fallback") != "forbidden"
        or not isinstance(required, dict)
        or not isinstance(source, dict)
    ):
        raise NativeRuntimeError("malformed native runtime lock: required contract fields differ")
    pins = cast("dict[object, object]", required)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in pins.items()):
        raise NativeRuntimeError("malformed native runtime lock: pins must be string pairs")
    typed_pins = cast("dict[str, str]", cast("object", pins))
    for package, expected in _EXPECTED.items():
        actual = typed_pins.get(package)
        if actual != expected:
            label = _DISPLAY_NAMES[package]
            raise NativeRuntimeError(f"{label} pin must be {expected}, found {actual}")
    if set(typed_pins) != set(_EXPECTED):
        raise NativeRuntimeError("malformed native runtime lock: unexpected runtime pins")
    typed_source = cast("dict[object, object]", source)
    if (
        typed_source.get("path") != "05_references/external_repos/pushT-so100/environment.yml"
        or typed_source.get("sha256")
        != "a7aab5a14bb18b6bb94cd1ecf13616384c6af87ba131ae5dc86fec7e94920f70"
        or typed_source.get("preservation") != "frozen_unchanged"
    ):
        raise NativeRuntimeError(
            "malformed native runtime lock: frozen environment identity differs"
        )
    return cast("NativeRuntimeLock", cast("object", raw))


def trusted_native_runtime_lock_digest(lock_path: Path = DEFAULT_LOCK) -> str:
    """Return the lock digest only after lock, sidecar, and workspace policy agree."""
    load_native_runtime_lock(lock_path)
    try:
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        sidecar = lock_path.with_name(f"{lock_path.name}.sha256")
        declared = sidecar.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as exc:
        raise NativeRuntimeError("native runtime lock digest sidecar is missing") from exc
    if declared != digest:
        raise NativeRuntimeError(
            f"native runtime lock digest mismatch: expected {declared}, found {digest}"
        )
    from .workspace import load_workspace_policy

    policy_digest = load_workspace_policy()["runtime"]["native_lock_sha256"]
    if policy_digest != digest:
        raise NativeRuntimeError(
            f"workspace native lock digest mismatch: expected {policy_digest}, found {digest}"
        )
    return digest


def detected_runtime() -> dict[str, str]:
    """Return exact versions at the native runtime boundary without importing frameworks."""
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for distribution in _EXPECTED:
        if distribution == "python":
            continue
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def assert_native_runtime(
    *,
    actual: Mapping[str, str] | None = None,
    lock: NativeRuntimeLock | None = None,
) -> None:
    """Reject every mismatch; this boundary intentionally has no fallback runtime."""
    selected_lock = load_native_runtime_lock() if lock is None else lock
    found = detected_runtime() if actual is None else dict(actual)
    mismatches = [
        f"{_DISPLAY_NAMES[name]}: expected {expected}, found {found.get(name, 'missing')}"
        for name, expected in selected_lock["required"].items()
        if found.get(name) != expected
    ]
    if mismatches:
        raise NativeRuntimeError(
            "native pushT-so100 runtime mismatch (fallback forbidden): " + "; ".join(mismatches)
        )


def native_runtime_report(
    *,
    actual: Mapping[str, str] | None = None,
    lock: NativeRuntimeLock | None = None,
    lock_path: Path = DEFAULT_LOCK,
    upstream_verifier: UpstreamVerifier = verify_pusht_so100,
) -> NativeRuntimeReport:
    """Validate and return governance, schema, lock identity, and detected versions."""
    selected_lock = load_native_runtime_lock(lock_path) if lock is None else lock
    found = detected_runtime() if actual is None else dict(actual)
    assert_native_runtime(actual=found, lock=selected_lock)
    digest = trusted_native_runtime_lock_digest(lock_path)
    source = selected_lock["source_environment"]
    source_path = PACKAGE_ROOT.parents[1] / source["path"]
    try:
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise NativeRuntimeError(
            f"frozen source environment is unavailable: {source_path}"
        ) from exc
    if source_digest != source["sha256"]:
        raise NativeRuntimeError(
            f"frozen source environment digest mismatch: expected {source['sha256']}, "
            f"found {source_digest}"
        )
    try:
        upstream = upstream_verifier(UPSTREAM_MANIFEST, UPSTREAM_ROOT)
    except UpstreamProvenanceError as exc:
        raise NativeRuntimeError(f"upstream provenance mismatch: {exc}") from exc
    return {
        "status": "compatible",
        "plan": PLAN,
        "contract_schema": CONTRACT_SCHEMA,
        "lock": _LOCK_RELATIVE,
        "lock_sha256": digest,
        "source_environment_sha256": source_digest,
        "fallback": "forbidden",
        "runtime": found,
        "upstream": upstream,
    }
