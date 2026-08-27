from __future__ import annotations

import os
from pathlib import Path

import numpy as np


FEATURES: dict[str, object] = {
    "observation.images.cam_top": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam_side": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (5,)},
    "action": {"dtype": "float32", "shape": (2,)},
}


def create_two_episode_repo(root: Path) -> Path:
    """Create a real local LeRobot 0.4.4 dataset with two unequal episodes."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        "local/todo4-native-fixture",
        fps=10,
        features=FEATURES,
        root=root,
        vcodec="h264",
    )
    for episode, length in enumerate((2, 3)):
        for frame in range(length):
            base = episode * 40 + frame * 3
            yy, xx = np.indices((224, 224), dtype=np.uint16)
            cam_top = np.empty((224, 224, 3), dtype=np.uint8)
            cam_top[:, :, 0] = (xx + base) % 256
            cam_top[:, :, 1] = (yy + base * 2) % 256
            cam_top[:, :, 2] = (xx + yy + base) % 256
            cam_side = np.empty((224, 224, 3), dtype=np.uint8)
            cam_side[:, :, 0] = (yy + 90 + base) % 256
            cam_side[:, :, 1] = (xx + 40 + base) % 256
            cam_side[:, :, 2] = (xx * 2 + base) % 256
            dataset.add_frame(
                {
                    "observation.images.cam_top": cam_top,
                    "observation.images.cam_side": cam_side,
                    "observation.state": np.asarray(
                        [episode, frame, episode + frame / 10, -frame, 0.25], dtype=np.float32
                    ),
                    "action": np.asarray(
                        [-0.5 + episode / 2 + frame / 10, 0.25 - frame / 20],
                        dtype=np.float32,
                    ),
                    "task": "pushT",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root
