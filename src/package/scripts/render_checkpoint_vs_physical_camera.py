"""Render the original checkpoint view beside the current physical crop."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from render_physical_camera_match import (
    load_camera_model,
    place_block_like_live_workspace,
    recolor_physical_workspace,
    render_camera,
)
from so101_pusht_benchmark.sim.physical_camera import fit_camera_frame


PROJECT_ROOT = Path("/home/intro/InternLab/02_InTro_Project")
PHYSICAL_FRAME = (
    PROJECT_ROOT / "04_experiments/camera_test/webcam_live_latest.jpg"
)
OUTPUT = (
    PROJECT_ROOT
    / "04_experiments/camera_test/checkpoint_sim_vs_physical_live.png"
)


def labelled(title: str, image: np.ndarray) -> np.ndarray:
    """Put one aspect-preserving camera frame under a readable title."""
    panel = np.full((526, 640, 3), (19, 30, 46), dtype=np.uint8)
    panel[46:] = fit_camera_frame(image, width=640, height=480)
    cv2.putText(
        panel,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (240, 245, 250),
        2,
    )
    return panel


def main() -> int:
    """Generate exactly two panels: checkpoint simulation and current camera."""
    model = load_camera_model()
    recolor_physical_workspace(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    place_block_like_live_workspace(model, data)
    mujoco.mj_forward(model, data)
    simulation = render_camera(
        model,
        data,
        "original_checkpoint_front",
    )
    physical_bgr = cv2.imread(str(PHYSICAL_FRAME), cv2.IMREAD_COLOR)
    if physical_bgr is None:
        raise RuntimeError(f"could not read {PHYSICAL_FRAME}")
    physical = cv2.cvtColor(physical_bgr, cv2.COLOR_BGR2RGB)
    comparison = np.concatenate(
        (
            labelled("SIM VIEW USED BY EXISTING CHECKPOINT", simulation),
            labelled("CURRENT PHYSICAL CAMERA CROP", physical),
        ),
        axis=1,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(OUTPUT),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError(f"could not write {OUTPUT}")
    print(f"Rendered {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
