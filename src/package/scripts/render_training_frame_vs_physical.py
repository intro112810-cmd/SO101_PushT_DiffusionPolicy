"""Place an actual stored training cam_top frame beside the live physical crop."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import zarr

from so101_pusht_benchmark.sim.physical_camera import fit_camera_frame


PROJECT_ROOT = Path("/home/intro/InternLab/02_InTro_Project")
TRAINING_DATASET = (
    PROJECT_ROOT
    / "04_experiments/so101_pusht_benchmark/datasets/"
    "frozen_four_model_200ep_s96"
)
PHYSICAL_FRAME = (
    PROJECT_ROOT / "04_experiments/camera_test/webcam_live_latest.jpg"
)
OUTPUT = (
    PROJECT_ROOT
    / "04_experiments/camera_test/checkpoint_sim_vs_physical_live.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-index", type=int, default=0)
    parser.add_argument(
        "--rotate-physical",
        choices=("none", "ccw", "cw", "180"),
        default="none",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def labelled(title: str, image_rgb: np.ndarray) -> np.ndarray:
    """Put one aspect-preserving RGB frame below a readable title."""
    panel = np.full((526, 640, 3), (19, 30, 46), dtype=np.uint8)
    panel[46:] = fit_camera_frame(image_rgb, width=640, height=480)
    cv2.putText(
        panel,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (240, 245, 250),
        2,
    )
    return panel


def main() -> int:
    args = parse_args()
    dataset = zarr.open_group(str(TRAINING_DATASET), mode="r")
    cam_top = dataset["data/cam_top"]
    if not 0 <= args.training_index < len(cam_top):
        raise ValueError(
            f"training index must be in 0..{len(cam_top) - 1}"
        )
    training_rgb = np.asarray(cam_top[args.training_index], dtype=np.uint8)
    if training_rgb.shape != (96, 96, 3):
        raise RuntimeError("stored cam_top must be uint8[96,96,3]")
    physical_bgr = cv2.imread(str(PHYSICAL_FRAME), cv2.IMREAD_COLOR)
    if physical_bgr is None:
        raise RuntimeError(f"could not read {PHYSICAL_FRAME}")
    physical_rgb = cv2.cvtColor(physical_bgr, cv2.COLOR_BGR2RGB)
    rotations = {
        "none": None,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rotation = rotations[args.rotate_physical]
    if rotation is not None:
        physical_rgb = cv2.rotate(physical_rgb, rotation)
    comparison = np.concatenate(
        (
            labelled(
                f"ACTUAL TRAINING cam_top | stored frame {args.training_index}",
                training_rgb,
            ),
            labelled("CURRENT PHYSICAL CAMERA CROP", physical_rgb),
        ),
        axis=1,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError(f"could not write {output}")
    print(
        f"Rendered {output} from actual dataset frame "
        f"{args.training_index}/{len(cam_top) - 1}; "
        f"physical_rotation={args.rotate_physical}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
