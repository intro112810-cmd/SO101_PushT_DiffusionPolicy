from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[3] / "06_tools_scripts/webcam_live_preview.py"


def test_task_crop_keeps_centerline_robot_and_task_region() -> None:
    specification = importlib.util.spec_from_file_location("webcam_live_preview", SCRIPT)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)

    cropped = module.crop_task_frame(frame)

    assert cropped.shape == (400, 400, 3)
    assert np.array_equal(cropped, frame[:400, 100:500])
