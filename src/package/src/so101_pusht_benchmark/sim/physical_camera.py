"""Camera poses used to compare frozen simulation geometry with the real setup."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


CALIBRATION_BLOCK_POSITION = (0.30, -0.08, 0.015)
UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class PhysicalCameraPose:
    """One static MuJoCo camera pose expressed in the human_env world frame."""

    name: str
    position: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy: float

    def xml(self) -> str:
        """Return this pose as a self-contained MuJoCo camera element."""
        position = " ".join(str(value) for value in self.position)
        xyaxes = " ".join(str(value) for value in self.xyaxes)
        return (
            f'<camera name="{self.name}" pos="{position}" xyaxes="{xyaxes}" '
            f'fovy="{self.fovy}"/>'
        )


def fit_camera_frame(
    image: UInt8Image,
    *,
    width: int,
    height: int,
) -> UInt8Image:
    """Letterbox one camera frame without changing its aspect ratio."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("camera frame must be uint8[H,W,3]")
    if width < 1 or height < 1:
        raise ValueError("output dimensions must be positive")
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


# The webcam is on the table-front centerline, looking back toward the base.
# It sees the T block before the arm and keeps the target between camera/robot.
ORIGINAL_CHECKPOINT_FRONT_CAMERA = PhysicalCameraPose(
    name="original_checkpoint_front",
    position=(0.55, -0.55, 0.35),
    xyaxes=(0.707, 0.707, 0.0, -0.331, 0.331, 0.884),
    fovy=45.0,
)


PHYSICAL_FRONT_CAMERA = PhysicalCameraPose(
    name="physical_front",
    position=(0.55, 0.0, 0.55),
    xyaxes=(0.0, 1.0, 0.0, -0.88, 0.0, 0.48),
    fovy=58.0,
)

# Retained only as a visual control: placing a camera behind the base occludes
# the task with the arm and does not match the physical reference frame.
BEHIND_ROBOT_CAMERA = PhysicalCameraPose(
    name="behind_robot",
    position=(-0.25, 0.0, 0.55),
    xyaxes=(0.0, -1.0, 0.0, 0.65, 0.0, 0.76),
    fovy=58.0,
)


def calibration_overlay_xml() -> str:
    """Return non-colliding table-rim and camera geometry for visual calibration."""
    return f"""
      {ORIGINAL_CHECKPOINT_FRONT_CAMERA.xml()}
      {PHYSICAL_FRONT_CAMERA.xml()}
      {BEHIND_ROBOT_CAMERA.xml()}
      <geom name="calibration_table_rim_left" type="box" pos=".25 .30 .003"
            size=".42 .008 .003" rgba=".12 .24 .46 1" contype="0" conaffinity="0"/>
      <geom name="calibration_table_rim_right" type="box" pos=".25 -.30 .003"
            size=".42 .008 .003" rgba=".12 .24 .46 1" contype="0" conaffinity="0"/>
      <geom name="calibration_table_rim_back" type="box" pos="-.17 0 .003"
            size=".008 .30 .003" rgba=".12 .24 .46 1" contype="0" conaffinity="0"/>
    """
