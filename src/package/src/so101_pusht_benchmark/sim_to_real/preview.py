"""Render predicted physical-inference actions inside an isolated MuJoCo rollout."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Protocol, cast

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..control.mouse_mapping import task_xy_to_pixels
from ..evaluation.frozen_env import load_frozen_pusht, validate_action


UInt8Image = NDArray[np.uint8]
BOUNDS_X = (0.15, 0.45)
BOUNDS_Y = (-0.15, 0.15)
FRAME_SIZE = 640


class _ImageIoRuntime(Protocol):
    def mimsave(
        self,
        uri: Path,
        images: Sequence[UInt8Image],
        **options: object,
    ) -> None: ...


def _image(observation: dict[str, NDArray[np.generic]]) -> UInt8Image:
    source = observation.get("_cam_top_hd", observation.get("cam_top"))
    if not isinstance(source, np.ndarray) or source.dtype != np.uint8:
        raise RuntimeError("MuJoCo preview requires a uint8 cam_top frame")
    return np.asarray(source, dtype=np.uint8)


def _annotate(
    image: UInt8Image,
    actions: Sequence[NDArray[np.float32]],
    active: int | None,
) -> UInt8Image:
    canvas = Image.fromarray(image).resize(
        (FRAME_SIZE, FRAME_SIZE),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(canvas)
    points: list[tuple[int, int]] = []
    for action in actions:
        px, py = task_xy_to_pixels(
            float(action[0]),
            float(action[1]),
            FRAME_SIZE,
            FRAME_SIZE,
            BOUNDS_X,
            BOUNDS_Y,
        )
        points.append((round(px), round(py)))
    for start, end in pairwise(points):
        draw.line((start, end), fill=(230, 30, 30), width=4)
    for index, point in enumerate(points):
        color = (30, 220, 30) if index == active else (20, 180, 255)
        draw.ellipse(
            (point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9),
            fill=color,
        )
        draw.text((point[0] + 10, point[1] - 8), str(index + 1), fill=color)
    draw.text(
        (18, 18),
        "MUJOCO DRY-RUN ONLY - NO PHYSICAL COMMAND",
        fill=(230, 20, 20),
    )
    return np.asarray(canvas, dtype=np.uint8)


def render_prediction_preview(
    predicted_actions: Sequence[Sequence[float]],
    *,
    png_path: Path,
    mp4_path: Path,
    seed: int,
    evidence_scope: str,
) -> dict[str, object]:
    """Simulate eight absolute-XY actions and persist visual-only proof."""
    if evidence_scope not in {"production", "test_fixture_only"}:
        raise ValueError("preview evidence scope is invalid")
    actions = [
        validate_action(np.asarray(action, dtype=np.float32)) for action in predicted_actions
    ]
    if len(actions) != 8:
        raise ValueError("MuJoCo preview requires exactly eight predicted actions")
    environment = load_frozen_pusht(max_steps=len(actions))
    frames: list[UInt8Image] = []
    try:
        observation, _ = environment.reset(seed=seed)
        frames.append(_annotate(_image(observation), actions, None))
        for index, action in enumerate(actions):
            step = environment.step(action)
            frames.append(_annotate(_image(step.observation), actions, index))
    finally:
        environment.close()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(frames[-1]).save(png_path)
        imageio_runtime = cast("_ImageIoRuntime", cast("object", imageio))
        imageio_runtime.mimsave(
            mp4_path,
            frames,
            fps=2,
            quality=8,
            codec="libx264",
            pixelformat="yuv420p",
        )
    except BaseException:
        for partial in (png_path, mp4_path):
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
        raise
    return {
        "preview_mode": "isolated_frozen_mujoco_trajectory",
        "evidence_scope": evidence_scope,
        "seed": seed,
        "predicted_target_count": len(actions),
        "frame_count": len(frames),
        "png": str(png_path.resolve()),
        "mp4": str(mp4_path.resolve()),
        "deployment_valid": False,
        "motor_writes_performed": False,
        "actuation_performed": False,
    }
