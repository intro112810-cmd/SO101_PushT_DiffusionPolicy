from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast

import numpy as np
import pytest

from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from so101_pusht_benchmark.data.splits import ExperimentConfig, freeze_training_view
from so101_pusht_benchmark.training.launcher import (
    MODEL_NAMES,
    resolved_config,
    smoke_probe_config,
    update_budget,
)
from so101_pusht_benchmark.training.model_smoke import validate_smoke_store
from so101_pusht_benchmark.workspace import runtime_artifact_root


def ineligible_fixture(root: Path, *, episodes: int = 1) -> Path:
    horizon = 16
    rows = horizon * episodes
    episode_ids = [f"fixture-episode-{index}" for index in range(episodes)]
    episode_ends = np.arange(horizon, rows + 1, horizon, dtype=np.int64)
    frame = np.tile(np.arange(horizon, dtype=np.int64), episodes)
    episode = np.repeat(np.arange(episodes, dtype=np.int64), horizon)
    single_action = np.stack(
        (
            np.linspace(-0.5, 0.5, horizon, dtype=np.float32),
            np.linspace(0.5, -0.5, horizon, dtype=np.float32),
        ),
        axis=1,
    )
    arrays = {
        "cam_top": PaperArray(np.zeros((rows, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "cam_side": PaperArray(np.ones((rows, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "agent_pos": PaperArray(np.zeros((rows, 5), dtype=np.float32), "radians"),
        "action": PaperArray(np.tile(single_action, (episodes, 1)), "absolute normalized mocap XY"),
        "timestamp": PaperArray(frame.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(episode, "episode ordinal"),
        "frame_index": PaperArray(frame, "frame ordinal"),
    }
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {"synthetic-fixture.bin": "7" * 64},
        "episodes": [{"episode_id": item, "length": horizon} for item in episode_ids],
    }
    return write_paper_view(
        root,
        arrays,
        episode_ends,
        PaperViewMetadata(
            canonical_digest(arrays, episode_ends, episode_ids),
            root_provenance_digest(provenance),
            provenance,
            episode_ids,
            {
                "frozen": False,
                "training_eligible": False,
                "reason": "synthetic_fixture_not_comparison_eligible",
                "train": [],
                "validation": [],
                "test": [],
            },
            trusted_runtime_lock_digest(),
            False,
        ),
    )


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_real_upstream_fixture_smoke_one_optimizer_step(
    canonical_test_root: Path, model: str
) -> None:
    root = canonical_test_root
    store = ineligible_fixture(root / "fixture")
    project_root = Path(__file__).resolve().parents[3]
    paper_python = (
        project_root / "04_experiments/so101_pusht_benchmark/cache/envs/paper-baselines/bin/python"
    )
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(project_root / "03_code/so101_pusht_benchmark/src"))
    driver = """
import json, sys
from so101_pusht_benchmark.training.model_smoke import run_one_batch_smoke
print('SMOKE_JSON=' + json.dumps(run_one_batch_smoke(sys.argv[1], sys.argv[2], mode='fixture'), sort_keys=True))
"""
    try:
        result = subprocess.run(
            [str(paper_python), "-c", driver, model, str(store)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        next(line for line in result.stdout.splitlines() if line.startswith("SMOKE_JSON=")).split(
            "=", 1
        )[1]
    )
    assert receipt["model"] == model
    assert receipt["mode"] == "fixture"
    assert receipt["training_eligible"] is False
    assert receipt["comparison_eligible"] is False
    assert receipt["action_dim"] == 2
    assert receipt["optimizer_steps"] == 1
    assert np.isfinite(receipt["loss"])
    assert "/cache/upstream/stanford/" in receipt["policy_origin"]
    assert "/cache/upstream/stanford/" in receipt["workspace_origin"]
    if model == "lstm_gmm":
        assert receipt["recurrent_identity"] == "BC_RNN_GMM/RNNGMMActorNetwork/LSTM/GMM"


def test_fixture_store_cannot_be_claimed_as_production(
    canonical_test_root: Path,
) -> None:
    root = canonical_test_root
    try:
        store = ineligible_fixture(root / "fixture")
        fixture = validate_smoke_store(store, mode="fixture")
        assert fixture == validate_smoke_store(store, mode="fixture")
        assert fixture.training_eligible is False
        assert fixture.comparison_eligible is False
        with pytest.raises(ValueError, match="production smoke requires immutable frozen manifest"):
            validate_smoke_store(store, mode="production")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_production_smoke_requires_and_binds_frozen_manifest(
    canonical_test_root: Path,
) -> None:
    root = canonical_test_root
    try:
        source = ineligible_fixture(root / "source", episodes=3)
        frozen, manifest = freeze_training_view(
            source,
            root / "frozen",
            ExperimentConfig(
                "pusht-so100-experiment-v1",
                3,
                {
                    "train": Decimal("0.34"),
                    "validation": Decimal("0.33"),
                    "test": Decimal("0.33"),
                },
            ),
        )
        production = validate_smoke_store(frozen, mode="production")
        assert production.training_eligible is True
        assert production.comparison_eligible is False
        assert production.split_digest == manifest.digest
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_smoke_probe_bounds_all_four_models() -> None:
    for model in MODEL_NAMES:
        config = resolved_config(model, runtime_artifact_root() / "paper-view", seed=0)
        probe = smoke_probe_config(config)
        training = cast("dict[str, object]", probe["training"])
        assert training["max_train_steps"] == 1
        assert training["num_epochs"] == 1
        assert update_budget(probe) == 1
        assert probe["_target_"] == config["_target_"]


def test_smoke_probe_keeps_original_classes_and_data() -> None:
    for model in MODEL_NAMES:
        config = resolved_config(model, runtime_artifact_root() / "paper-view", seed=1)
        probe = smoke_probe_config(config)
        task = cast("dict[str, object]", probe["task"])
        dataset = cast("dict[str, object]", task["dataset"])
        assert dataset["_target_"] == (
            "so101_pusht_benchmark.integrations.paper_baselines.dataset.PaperBaselineDataset"
        )
        # The smoke probe uses the adapter's ineligible-probe split so a
        # 1-episode store can be trained; the real policy/data classes stay.
        assert dataset["split"] == "synthetic_probe"
        policy = cast("dict[str, object]", probe["policy"])
        original_policy = cast("dict[str, object]", config["policy"])
        assert policy["_target_"] == original_policy["_target_"]
