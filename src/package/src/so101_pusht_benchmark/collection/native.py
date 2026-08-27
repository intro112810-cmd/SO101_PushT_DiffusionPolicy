"""Fail-closed launcher for the frozen pushT-so100 F710 collector."""

from __future__ import annotations

import importlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from ..core.upstream_provenance import UpstreamProvenanceError, verify_pusht_so100
from ..native_runtime import native_runtime_report
from ..workspace import runtime_artifact_root


PACKAGE_ROOT = Path(__file__).parents[3]
UPSTREAM_ROOT = PACKAGE_ROOT.parents[1] / "05_references/external_repos/pushT-so100"
UPSTREAM_ENTRYPOINT = Path("src/env_human_ee.py")
UPSTREAM_MANIFEST = PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json"
DEVICE_PROBE_ENV = "SO101_PUSHT_F710_DEVICES_JSON"
DISPLAY_PROBE_ENV = "SO101_PUSHT_DISPLAY_AVAILABLE"
F710_NAME = "Logitech Gamepad F710"
F710_MIN_AXES = 4
F710_MIN_BUTTONS = 8
FROZEN_FPS = 10
FROZEN_MOVE_SPEED = 0.05
FROZEN_ROTATION_SPEED = 1.0

RuntimeReporter = Callable[[], Mapping[str, object]]
DeviceProbe = Callable[[], Sequence["F710Device"]]
ProcessRunner = Callable[[tuple[str, ...], Path, dict[str, str]], int]
ProvenanceVerifier = Callable[[], Mapping[str, object]]

_ALLOWED_OPERATOR_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
    "LANG",
    "LC_ALL",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_RUNTIME_DIR",
)
_DETERMINISTIC_RUNTIME_ENV = {
    "MUJOCO_GL": "egl",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
_TRUSTED_CHILD_PATH = f"{Path(sys.executable).parent}:/usr/bin:/bin"
_PYGAME_PKG_RESOURCES_WARNING = (
    r"^pkg_resources is deprecated as an API\. See "
    r"https://setuptools\.pypa\.io/en/latest/pkg_resources\.html\. The pkg_resources "
    r"package is slated for removal as early as 2025-11-30\. Refrain from using "
    r"this package or pin to Setuptools<81\.$"
)


class _Joystick(Protocol):
    def get_name(self) -> str: ...

    def get_numaxes(self) -> int: ...

    def get_numbuttons(self) -> int: ...


class _JoystickModule(Protocol):
    def init(self) -> None: ...

    def quit(self) -> None: ...

    def get_count(self) -> int: ...

    Joystick: Callable[[int], _Joystick]


class _Pygame(Protocol):
    joystick: _JoystickModule


class NativeCollectionError(RuntimeError):
    """Raised when native collection cannot preserve the frozen runtime contract."""


@dataclass(frozen=True)
class F710Device:
    """The device identity and capabilities consumed by the frozen mapping."""

    name: str
    axes: int
    buttons: int


@dataclass(frozen=True)
class NativeCollectionRequest:
    """Validated arguments accepted by the frozen upstream entrypoint."""

    dataset_root: Path
    fps: int = FROZEN_FPS
    move_speed: float = FROZEN_MOVE_SPEED
    rotation_speed: float = FROZEN_ROTATION_SPEED


@dataclass(frozen=True)
class NativeCollectionPlan:
    """Exact subprocess invocation produced by native collection preflight."""

    cwd: Path
    argv: tuple[str, ...]
    environment: dict[str, str]
    dataset_root: Path
    allowed_dataset_root: Path
    device: F710Device
    runtime: Mapping[str, object]

    def report(self) -> dict[str, object]:
        """Return deterministic, non-secret launch details for CLI output."""
        display_environment = {
            name: self.environment[name]
            for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "MUJOCO_GL")
            if name in self.environment
        }
        return {
            "status": "ready",
            "command": "collect-native",
            "adapter": "frozen_pushT_so100",
            "mapping": {
                "axes": {"x": 0, "y": 1, "rotation": 3},
                "buttons": {
                    "z_up": 4,
                    "z_down": 0,
                    "reset": 3,
                    "record_toggle": 1,
                    "exit": 7,
                },
                "deadzone": 0.1,
                "move_speed": FROZEN_MOVE_SPEED,
                "rotation_speed": FROZEN_ROTATION_SPEED,
                "button_debounce_seconds": 0.3,
            },
            "device": {
                "name": self.device.name,
                "axes": self.device.axes,
                "buttons": self.device.buttons,
            },
            "cwd": str(self.cwd),
            "upstream_entrypoint": str(UPSTREAM_ENTRYPOINT),
            "argv": list(self.argv),
            "environment": display_environment,
            "runtime": dict(self.runtime),
        }


def _injected_devices(environment: Mapping[str, str]) -> tuple[F710Device, ...] | None:
    encoded = environment.get(DEVICE_PROBE_ENV)
    if encoded is None:
        return None
    try:
        value: object = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise NativeCollectionError("malformed injected joystick probe") from exc
    if not isinstance(value, list):
        raise NativeCollectionError("malformed injected joystick probe")
    devices: list[F710Device] = []
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            raise NativeCollectionError("malformed injected joystick probe")
        raw = cast("dict[object, object]", item)
        if set(raw) != {"name", "axes", "buttons"}:
            raise NativeCollectionError("malformed injected joystick probe")
        name = raw["name"]
        axes = raw["axes"]
        buttons = raw["buttons"]
        if (
            not isinstance(name, str)
            or type(axes) is not int
            or type(buttons) is not int
            or axes < 0
            or buttons < 0
        ):
            raise NativeCollectionError("malformed injected joystick probe")
        devices.append(F710Device(name=name, axes=axes, buttons=buttons))
    return tuple(devices)


def probe_joysticks(environment: Mapping[str, str]) -> tuple[F710Device, ...]:
    """Enumerate pygame joysticks, or consume the explicit process-level test seam."""
    injected = _injected_devices(environment)
    if injected is not None:
        return injected
    previous_prompt = os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT")
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_PYGAME_PKG_RESOURCES_WARNING,
                category=UserWarning,
                module=r"^pygame\.pkgdata$",
            )
            pygame = cast("_Pygame", cast("object", importlib.import_module("pygame")))
        pygame.joystick.init()
        try:
            found: list[F710Device] = []
            for index in range(pygame.joystick.get_count()):
                joystick = pygame.joystick.Joystick(index)
                found.append(
                    F710Device(
                        name=joystick.get_name(),
                        axes=joystick.get_numaxes(),
                        buttons=joystick.get_numbuttons(),
                    )
                )
            devices = tuple(found)
        finally:
            pygame.joystick.quit()
    except Exception as exc:
        raise NativeCollectionError("F710 joystick unavailable") from exc
    finally:
        if previous_prompt is None:
            os.environ.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)
        else:
            os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = previous_prompt
    return devices


def display_available(environment: Mapping[str, str]) -> bool:
    """Check that the selected X11 or Wayland display socket exists."""
    injected = environment.get(DISPLAY_PROBE_ENV)
    if injected is not None:
        if injected not in {"0", "1"}:
            raise NativeCollectionError("malformed injected display probe")
        return injected == "1"
    display = environment.get("DISPLAY")
    if display:
        display_number = display.removeprefix(":").split(".", maxsplit=1)[0]
        x11_socket = Path(tempfile.gettempdir()) / ".X11-unix" / f"X{display_number}"
        if display_number.isdigit() and x11_socket.is_socket():
            return True
    wayland_display = environment.get("WAYLAND_DISPLAY")
    runtime_dir = environment.get("XDG_RUNTIME_DIR")
    return bool(
        wayland_display and runtime_dir and (Path(runtime_dir) / wayland_display).is_socket()
    )


def _absolute_lexical(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise NativeCollectionError(f"dataset path unavailable: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise NativeCollectionError(f"symlink dataset path is forbidden: {path}")
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise NativeCollectionError(f"dataset parent unavailable: {current}")


def _reject_dataset_tree_symlinks(dataset_root: Path) -> None:
    pending = [dataset_root]
    while pending:
        current = pending.pop()
        try:
            entries = tuple(os.scandir(current))
        except OSError as exc:
            raise NativeCollectionError(f"dataset tree unavailable: {current}") from exc
        for entry in entries:
            if entry.is_symlink():
                raise NativeCollectionError(
                    f"dataset tree contains symlink: {Path(entry.path)}"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))


def _validate_dataset_path(dataset_path: Path, allowed_root: Path) -> tuple[Path, Path]:
    lexical_root = _absolute_lexical(allowed_root)
    lexical_dataset = _absolute_lexical(dataset_path)
    if ".." in allowed_root.parts or ".." in dataset_path.parts:
        raise NativeCollectionError("dataset root path traversal is forbidden")
    _reject_symlink_components(lexical_root)
    _reject_symlink_components(lexical_dataset)
    try:
        canonical_root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise NativeCollectionError(f"allowed dataset root unavailable: {lexical_root}") from exc
    if not canonical_root.is_dir():
        raise NativeCollectionError(f"allowed dataset root unavailable: {lexical_root}")
    canonical_dataset = lexical_dataset.resolve(strict=False)
    try:
        canonical_dataset.relative_to(canonical_root)
    except ValueError as exc:
        raise NativeCollectionError(
            f"dataset path is outside canonical dataset root: {lexical_dataset}"
        ) from exc
    if canonical_dataset == canonical_root:
        raise NativeCollectionError("dataset path must be beneath canonical dataset root")
    if canonical_dataset.exists() and not canonical_dataset.is_dir():
        raise NativeCollectionError(f"dataset path is not a directory: {canonical_dataset}")
    if canonical_dataset.exists():
        _reject_dataset_tree_symlinks(canonical_dataset)
    parent = canonical_dataset.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise NativeCollectionError(f"dataset parent unavailable: {parent}")
    return canonical_dataset, canonical_root


def _child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    selected = {name: environment[name] for name in _ALLOWED_OPERATOR_ENV if name in environment}
    selected.update(_DETERMINISTIC_RUNTIME_ENV)
    selected["PATH"] = _TRUSTED_CHILD_PATH
    return selected


def _strict_upstream_provenance() -> Mapping[str, object]:
    try:
        return verify_pusht_so100(UPSTREAM_MANIFEST, UPSTREAM_ROOT)
    except UpstreamProvenanceError as exc:
        raise NativeCollectionError(str(exc)) from exc


def _validate_request(
    request: NativeCollectionRequest,
    allowed_dataset_root: Path,
) -> tuple[Path, Path]:
    if type(request.fps) is not int or request.fps != FROZEN_FPS:
        raise NativeCollectionError("FPS must be exactly 10")
    if (
        type(request.move_speed) is not float
        or not math.isfinite(request.move_speed)
        or request.move_speed != FROZEN_MOVE_SPEED
    ):
        raise NativeCollectionError("move speed must be exactly 0.05")
    if (
        type(request.rotation_speed) is not float
        or not math.isfinite(request.rotation_speed)
        or request.rotation_speed != FROZEN_ROTATION_SPEED
    ):
        raise NativeCollectionError("rotation speed must be exactly 1.0")
    return _validate_dataset_path(request.dataset_root, allowed_dataset_root)


def preflight_native_collection(
    request: NativeCollectionRequest,
    *,
    runtime_report: RuntimeReporter = native_runtime_report,
    device_probe: DeviceProbe | None = None,
    environment: Mapping[str, str] | None = None,
    executable: str | None = None,
    allowed_dataset_root: Path | None = None,
) -> NativeCollectionPlan:
    """Validate all boundaries without creating the dataset or starting MuJoCo."""
    selected_allowed_root = (
        runtime_artifact_root() / "datasets"
        if allowed_dataset_root is None
        else allowed_dataset_root
    )
    dataset_root, canonical_allowed_root = _validate_request(request, selected_allowed_root)
    selected_environment = dict(os.environ if environment is None else environment)
    runtime = runtime_report()
    devices = (
        tuple(device_probe()) if device_probe is not None else probe_joysticks(selected_environment)
    )
    def is_capable(device: F710Device) -> bool:
        return (
            device.name == F710_NAME
            and device.axes >= F710_MIN_AXES
            and device.buttons >= F710_MIN_BUTTONS
        )

    capable = devices[0] if devices and is_capable(devices[0]) else None
    if capable is None and any(is_capable(device) for device in devices[1:]):
        raise NativeCollectionError(
            "joystick index 0 must be a capable Logitech Gamepad F710"
        )
    if capable is None:
        raise NativeCollectionError("F710 joystick unavailable")
    if not display_available(selected_environment):
        raise NativeCollectionError("graphical display unavailable")
    child_environment = _child_environment(selected_environment)
    python = sys.executable if executable is None else executable
    return NativeCollectionPlan(
        cwd=UPSTREAM_ROOT,
        argv=(
            python,
            str(UPSTREAM_ENTRYPOINT),
            "--repo_id",
            str(dataset_root),
            "--fps",
            str(request.fps),
            "--move_speed",
            str(request.move_speed),
            "--rot_speed",
            str(request.rotation_speed),
        ),
        environment=child_environment,
        dataset_root=dataset_root,
        allowed_dataset_root=canonical_allowed_root,
        device=capable,
        runtime=runtime,
    )


def _run_process(argv: tuple[str, ...], cwd: Path, environment: dict[str, str]) -> int:
    completed = subprocess.run(list(argv), cwd=cwd, env=environment, check=False)
    return completed.returncode


def launch_native_collection(
    plan: NativeCollectionPlan,
    *,
    runner: ProcessRunner = _run_process,
    provenance_verifier: ProvenanceVerifier = _strict_upstream_provenance,
) -> int:
    """Revalidate every boundary, verify upstream bytes, then launch the collector."""
    dataset_root, allowed_root = _validate_dataset_path(
        plan.dataset_root, plan.allowed_dataset_root
    )
    expected_argv = (
        plan.argv[0],
        str(UPSTREAM_ENTRYPOINT),
        "--repo_id",
        str(dataset_root),
        "--fps",
        str(FROZEN_FPS),
        "--move_speed",
        str(FROZEN_MOVE_SPEED),
        "--rot_speed",
        str(FROZEN_ROTATION_SPEED),
    )
    if (
        dataset_root != plan.dataset_root
        or allowed_root != plan.allowed_dataset_root
        or plan.cwd != UPSTREAM_ROOT
        or plan.argv != expected_argv
        or plan.environment != _child_environment(plan.environment)
    ):
        raise NativeCollectionError("native collection plan changed after preflight")
    provenance_verifier()
    return runner(plan.argv, plan.cwd, dict(plan.environment))
