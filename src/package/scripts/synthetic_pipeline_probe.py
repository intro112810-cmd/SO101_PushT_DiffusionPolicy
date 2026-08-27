"""Temporary developer probe; outputs are ineligible and deleted after verification."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from so101_pusht_benchmark.data.paper_view import PaperArray, PaperViewMetadata, write_paper_view
from so101_pusht_benchmark.training.artifacts import ArtifactIndex
from so101_pusht_benchmark.training.evaluator import EvaluationRequest, evaluate_bundle
from so101_pusht_benchmark.training.exporter import export_inference_bundle
from so101_pusht_benchmark.training.launcher import TrainingLaunch, launch_training
from so101_pusht_benchmark.workspace import runtime_artifact_root


def _fixture(root: Path) -> Path:
    episodes, frames_per_episode = 200, 8
    count = episodes * frames_per_episode
    episode_ids = [f"synthetic-{index:03d}" for index in range(episodes)]
    time = np.arange(count, dtype=np.float32)
    front = np.zeros((count, 96, 96, 3), dtype=np.uint8)
    front[:, 40:56, 40:56, 0] = (time.astype(np.uint16) % 255).astype(np.uint8)[:, None, None]
    state = np.stack([np.sin(time / (20 + index)) for index in range(6)], axis=1).astype(np.float32)
    action = np.stack((0.28 + 0.02 * np.sin(time / 10), 0.02 * np.cos(time / 10)), axis=1)
    arrays = {
        "front": PaperArray(front, "rgb intensity"),
        "state": PaperArray(state, "radians"),
        "action": PaperArray(action.astype(np.float32), "meters"),
        "timestamp": PaperArray(np.arange(count, dtype=np.float64) / 10, "seconds"),
        "episode_id": PaperArray(
            np.repeat(np.arange(episodes), frames_per_episode), "episode ordinal"
        ),
        "frame_index": PaperArray(
            np.tile(np.arange(frames_per_episode), episodes), "frame ordinal"
        ),
    }
    split: dict[str, object] = {
        "frozen": True,
        "training_eligible": False,
        "train": episode_ids[:160],
        "validation": episode_ids[160:180],
        "test": episode_ids[180:],
    }
    metadata = PaperViewMetadata(
        "synthetic-pipeline-probe",
        "synthetic-ineligible",
        episode_ids,
        split,
        "0" * 64,
        False,
    )
    ends = np.arange(1, episodes + 1, dtype=np.int64) * frames_per_episode
    return write_paper_view(root / "view", arrays, ends, metadata)


def main() -> int:
    artifact_root = runtime_artifact_root()
    with TemporaryDirectory(prefix="dp-cnn-probe-", dir=artifact_root) as temporary:
        root = Path(temporary)
        index_path = root / "artifact_index.json"
        index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
        index = ArtifactIndex(index_path, root)
        view = _fixture(root)
        checkpoint = launch_training(
            view,
            root / "run",
            index,
            TrainingLaunch(seed=0, artifact_id="synthetic-probe", simulation_probe=True),
        )
        bundle = export_inference_bundle(
            checkpoint,
            root / "run/resolved_config.json",
            root / "bundle",
            artifact_id="synthetic-probe",
            index=index,
        )
        metrics = evaluate_bundle(
            bundle,
            root / "evaluation",
            index,
            EvaluationRequest("synthetic-probe", seeds=(100000,), max_steps=1),
        )
        logs = [json.loads(line) for line in (root / "run/logs.json.txt").read_text().splitlines()]
        losses = [float(item["train_loss"]) for item in logs if "val_loss" in item]
        result = {
            "deployment_scope": "simulation_only",
            "training_eligible": False,
            "checkpoint": checkpoint.is_file(),
            "bundle": bundle.is_file(),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_reduced": losses[-1] < losses[0],
            "rollout": bool((root / "evaluation/videos/seed-100000.mp4").is_file()),
            "evaluation": metrics.is_file(),
            "temporary_outputs_deleted": True,
        }
        print(json.dumps(result, sort_keys=True))
        if not all(
            (
                result["checkpoint"],
                result["bundle"],
                result["rollout"],
                result["evaluation"],
                result["loss_reduced"],
            )
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
