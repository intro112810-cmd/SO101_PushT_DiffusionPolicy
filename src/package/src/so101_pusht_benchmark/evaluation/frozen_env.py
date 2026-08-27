"""Verified adapter for the frozen upstream ``env_gym_ee.PushT`` environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
import importlib.util
import math
import sys
from typing import Protocol, SupportsFloat, cast

import numpy as np
from numpy.typing import NDArray

from ..core.upstream_provenance import verify_pusht_so100
from ..workspace import PACKAGE_ROOT, PROJECT_ROOT

JOINT_ORDER = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
_UPSTREAM_ROOT = PROJECT_ROOT / "05_references/external_repos/pushT-so100"
_MANIFEST = PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json"
_XML = _UPSTREAM_ROOT / "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"


class ActionContractError(ValueError):
    """Raised before stepping when an action violates the native action contract."""


class FrozenEnvironmentError(RuntimeError):
    """Raised when frozen environment bytes or observations violate their contract."""


@dataclass(frozen=True, slots=True)
class FrozenStep:
    observation: dict[str, NDArray[np.generic]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class _RawEnvironment(Protocol):
    def reset(self, seed: int | None = None) -> tuple[object, object]: ...

    def step(self, action: object) -> tuple[object, object, object, object, object]: ...

    def close(self) -> None: ...


def validate_action(value: object) -> NDArray[np.float32]:
    """Require the exact native dtype, shape, finite values, and bounds without coercion."""
    if not isinstance(value, np.ndarray):
        raise ActionContractError("action must be exact float32[2]")
    typed_value = cast("NDArray[np.generic]", cast("object", value))
    if typed_value.dtype != np.dtype(np.float32):
        raise ActionContractError("action must be exact float32[2]")
    action = cast("NDArray[np.float32]", cast("object", typed_value))
    if action.shape != (2,):
        raise ActionContractError("action must be exact float32[2]")
    if not bool(np.isfinite(action).all()):
        raise ActionContractError("action must contain only finite values")
    if bool(np.any(action < -1.0)) or bool(np.any(action > 1.0)):
        raise ActionContractError("action must remain within [-1,1] bounds; clipping is forbidden")
    return action


def _observation(value: object) -> dict[str, NDArray[np.generic]]:
    if not isinstance(value, dict):
        raise FrozenEnvironmentError("frozen observation must be a mapping")
    raw = cast("dict[str, object]", value)
    single_cam = os.environ.get("PUSHT_SINGLE_CAM") == "1"
    expected = ("cam_top", "cam_side", *JOINT_ORDER)
    if tuple(raw) != expected:
        raise FrozenEnvironmentError("frozen observation keys/order mismatch")
    result: dict[str, NDArray[np.generic]] = {}
    if single_cam:
        # Local exploratory policies consume only cam_top downscaled to 96x96.
        image = raw["cam_top"]
        if not isinstance(image, np.ndarray):
            raise FrozenEnvironmentError("cam_top must be uint8[224,224,3]")
        typed_image = cast("NDArray[np.generic]", cast("object", image))
        if typed_image.shape != (224, 224, 3) or typed_image.dtype != np.dtype(np.uint8):
            raise FrozenEnvironmentError("cam_top must be uint8[224,224,3]")
        import cv2

        resized = cv2.resize(
            typed_image,
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        result["cam_top"] = np.asarray(resized, dtype=np.uint8)
        result["_cam_top_hd"] = np.asarray(typed_image, dtype=np.uint8)
    else:
        for camera in ("cam_top", "cam_side"):
            image = raw[camera]
            if not isinstance(image, np.ndarray):
                raise FrozenEnvironmentError(f"{camera} must be uint8[224,224,3]")
            typed_image = cast("NDArray[np.generic]", cast("object", image))
            if typed_image.shape != (224, 224, 3) or typed_image.dtype != np.dtype(np.uint8):
                raise FrozenEnvironmentError(f"{camera} must be uint8[224,224,3]")
            result[camera] = typed_image
    joints: list[float] = []
    for name in JOINT_ORDER:
        item = raw[name]
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise FrozenEnvironmentError(f"joint {name} must be a finite scalar")
        number = float(cast("SupportsFloat", item))
        if not math.isfinite(number):
            raise FrozenEnvironmentError(f"joint {name} must be a finite scalar")
        joints.append(number)
    result["agent_pos"] = np.asarray(joints, dtype=np.float32)
    return result


class FrozenPushTAdapter:
    """Expose only the canonical native policy contract around frozen core behavior."""

    def __init__(self, environment: object) -> None:
        for member in ("reset", "step", "close"):
            if not callable(getattr(environment, member, None)):
                raise FrozenEnvironmentError("frozen PushT environment lifecycle is incomplete")
        self._environment = cast("_RawEnvironment", environment)
        self._closed = False

    @property
    def raw_environment(self) -> _RawEnvironment:
        """Expose the verified upstream instance for read-only visualization."""
        return self._environment

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        raw_observation, raw_info = self._environment.reset(seed=seed)
        if not isinstance(raw_info, dict):
            raise FrozenEnvironmentError("frozen reset info must be a mapping")
        return _observation(raw_observation), cast("dict[str, object]", raw_info)

    def step(self, action: object) -> FrozenStep:
        checked = validate_action(action)
        raw_observation, reward, terminated, truncated, raw_info = self._environment.step(checked)
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float, np.integer, np.floating))
            or not math.isfinite(float(cast("SupportsFloat", reward)))
            or not isinstance(terminated, (bool, np.bool_))
            or not isinstance(truncated, (bool, np.bool_))
            or not isinstance(raw_info, dict)
        ):
            raise FrozenEnvironmentError("frozen step result is malformed")
        info = cast("dict[str, object]", raw_info)
        if tuple(info) != ("dxy", "dyaw"):
            raise FrozenEnvironmentError("frozen step telemetry must be exactly dxy/dyaw")
        for key in ("dxy", "dyaw"):
            metric = info[key]
            if isinstance(metric, bool) or not isinstance(metric, (int, float, np.floating)):
                raise FrozenEnvironmentError(f"{key} must be finite")
            if not math.isfinite(float(cast("SupportsFloat", metric))):
                raise FrozenEnvironmentError(f"{key} must be finite")
        return FrozenStep(
            _observation(raw_observation),
            float(cast("SupportsFloat", reward)),
            bool(cast("bool", terminated)),
            bool(cast("bool", truncated)),
            {
                "dxy": float(cast("SupportsFloat", info["dxy"])),
                "dyaw": float(cast("SupportsFloat", info["dyaw"])),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        renderer = getattr(self._environment, "renderer", None)
        close_renderer = getattr(renderer, "close", None)
        try:
            self._environment.close()
        finally:
            if callable(close_renderer):
                close_renderer()


def load_frozen_pusht(*, max_steps: int = 300) -> FrozenPushTAdapter:
    """Verify every runtime byte, load frozen PushT by file, and return the adapter."""
    if type(max_steps) is not int or max_steps < 1:
        raise FrozenEnvironmentError("max_steps must be a positive integer")
    verify_pusht_so100(_MANIFEST, _UPSTREAM_ROOT)
    source = _UPSTREAM_ROOT / "src/env_gym_ee.py"
    spec = importlib.util.spec_from_file_location("_pusht_so100_frozen_env_gym_ee", source)
    if spec is None or spec.loader is None:
        raise FrozenEnvironmentError("cannot load frozen env_gym_ee.py")
    module = importlib.util.module_from_spec(spec)
    helper_source = source.parent / "helper.py"
    helper_spec = importlib.util.spec_from_file_location("helper", helper_source)
    if helper_spec is None or helper_spec.loader is None:
        raise FrozenEnvironmentError("cannot load frozen helper.py")
    helper_module = importlib.util.module_from_spec(helper_spec)
    previous_helper = sys.modules.get("helper")
    try:
        sys.modules["helper"] = helper_module
        helper_spec.loader.exec_module(helper_module)
        spec.loader.exec_module(module)
    finally:
        if previous_helper is None:
            sys.modules.pop("helper", None)
        else:
            sys.modules["helper"] = previous_helper
    environment_class = getattr(module, "PushT", None)
    if not isinstance(environment_class, type) or environment_class.__module__ != module.__name__:
        raise FrozenEnvironmentError("frozen PushT class origin is invalid")
    environment = environment_class(
        xml_path=str(_XML), max_steps=max_steps, render_mode="rgb_array"
    )
    return FrozenPushTAdapter(environment)
