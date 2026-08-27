from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
import inspect
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.policy.ibc_dfo_hybrid_image_policy import IbcDfoHybridImagePolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.workspace.train_robomimic_image_workspace import (
    TrainRobomimicImageWorkspace,
)

from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from so101_pusht_benchmark.data.splits import ExperimentConfig, freeze_training_view
from so101_pusht_benchmark.integrations.paper_baselines.configs import PROFILES, workspace_config
from so101_pusht_benchmark.integrations.paper_baselines.dataset import PaperBaselineDataset
from so101_pusht_benchmark.integrations.paper_baselines.runner import (
    PaperBaselineRunner,
    policy_seed,
)
from so101_pusht_benchmark.sim.env import StepResult


def _view(
    root: Path,
    *,
    unit: str = "absolute normalized mocap XY",
    training_eligible: bool = True,
    bad_action: bool = False,
    bad_split: bool = False,
) -> Path:
    count, length = 10, 3
    rows = count * length
    episode_ids = [f"episode-{index:02d}" for index in range(count)]
    episode = np.repeat(np.arange(count, dtype=np.int64), length)
    frame = np.tile(np.arange(length, dtype=np.int64), count)
    action = np.stack((episode / 20, -episode / 20), axis=1).astype(np.float32)
    if bad_action:
        action[0, 0] = 2.0
    arrays = {
        "cam_top": PaperArray(np.zeros((rows, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "cam_side": PaperArray(np.ones((rows, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "agent_pos": PaperArray(
            np.repeat(episode[:, None], 5, axis=1).astype(np.float32), "radians"
        ),
        "action": PaperArray(action, unit),
        "timestamp": PaperArray(frame.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(episode, "episode ordinal"),
        "frame_index": PaperArray(frame, "frame ordinal"),
    }
    ends = np.arange(length, rows + 1, length, dtype=np.int64)
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {"fixture.bin": "3" * 64},
        "episodes": [{"episode_id": item, "length": length} for item in episode_ids],
    }
    source = write_paper_view(
        root / "source",
        arrays,
        ends,
        PaperViewMetadata(
            canonical_digest(arrays, ends, episode_ids),
            root_provenance_digest(provenance),
            provenance,
            episode_ids,
            {
                "frozen": False,
                "training_eligible": False,
                "reason": "split_manifest_not_frozen",
                "train": [],
                "validation": [],
                "test": [],
            },
            trusted_runtime_lock_digest(),
            False,
        ),
    )
    if not training_eligible:
        return source
    config = ExperimentConfig(
        "pusht-so100-experiment-v1",
        count,
        {
            "train": Decimal("0.8"),
            "validation": Decimal("0.1"),
            "test": Decimal("0.1"),
        },
    )
    frozen, _ = freeze_training_view(source, root / "frozen", config)
    if bad_split:
        split_path = frozen / "splits.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["splits"]["validation"][0] = split["splits"]["train"][0]
        split_path.write_text(json.dumps(split), encoding="utf-8")
        manifest_path = frozen / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["splits"] = split
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return frozen


def test_dataset_exact_splits_dtypes_horizon_and_padding(
    canonical_test_root: Path,
) -> None:
    fixture = canonical_test_root / "view"
    import shutil

    try:
        dataset = PaperBaselineDataset(_view(fixture), horizon=3, pad_before=1, pad_after=1)
        assert len(dataset) == 24
        assert dataset.episode_ids == tuple(f"episode-{index:02d}" for index in range(8))
        sample = dataset.sample_window(0)
        sample_obs = cast("dict[str, torch.Tensor]", sample["obs"])
        sample_action = cast("torch.Tensor", sample["action"])
        assert sample_obs["cam_top"].shape == (3, 3, 224, 224)
        assert sample_obs["cam_side"].shape == (3, 3, 224, 224)
        assert sample_obs["agent_pos"].shape == (3, 5)
        assert sample_action.shape == (3, 2)
        assert torch.equal(sample_action[0], sample_action[1])
        assert dataset.get_validation_dataset().episode_ids == ("episode-08",)
        assert dataset.get_test_dataset().episode_ids == ("episode-09",)
    finally:
        shutil.rmtree(fixture, ignore_errors=True)


def test_dataset_rejects_wrong_unit_split_and_training_status(
    canonical_test_root: Path,
) -> None:
    import shutil

    cases = (
        ("unit", "radians", True, False, False, "unit"),
        ("status", "absolute normalized mocap XY", False, False, False, "training eligible"),
        ("action", "absolute normalized mocap XY", True, True, False, r"\[-1,1\]"),
        ("split", "absolute normalized mocap XY", True, False, True, "overlap"),
    )
    for suffix, unit, eligible, bad_action, bad_split, match in cases:
        root = canonical_test_root / suffix
        try:
            with pytest.raises(ValueError, match=match):
                PaperBaselineDataset(
                    _view(
                        root,
                        unit=unit,
                        training_eligible=eligible,
                        bad_action=bad_action,
                        bad_split=bad_split,
                    ),
                    horizon=2,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)


def test_normalizer_uses_train_rows_only(canonical_test_root: Path) -> None:
    import shutil

    root = canonical_test_root / "normalizer"
    try:
        dataset = PaperBaselineDataset(_view(root), horizon=2, pad_after=1)
        stats = dataset.get_normalizer().get_input_stats()
        assert torch.allclose(stats["action"]["max"], torch.tensor([0.35, 0.0]))
        assert torch.equal(stats["agent_pos"]["max"], torch.full((5,), 7.0))
        assert dataset.get_normalizer()["cam_top"] is not None
        assert dataset.get_normalizer()["cam_side"] is not None
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _FakeEnv:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.actions: list[list[float]] = []
        self.tick = 0

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        assert seed is not None
        self.seeds.append(seed)
        self.tick = 0
        return _observation(), {"seed": seed}

    def step(self, action: object) -> StepResult:
        value = cast("NDArray[np.float32]", action)
        self.actions.append(value.tolist())
        self.tick += 1
        return StepResult(
            _observation(),
            0.0,
            self.tick == 2,
            False,
            {"dxy": 0.02, "dyaw": 3.0},
        )

    def close(self) -> None:
        return None


def _observation() -> dict[str, NDArray[np.generic]]:
    return {
        "cam_top": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_side": np.zeros((224, 224, 3), dtype=np.uint8),
        "agent_pos": np.zeros(5, dtype=np.float32),
    }


class _Policy(BaseImagePolicy):
    def __init__(self) -> None:
        module_init = cast("Callable[[torch.nn.Module], None]", torch.nn.Module.__init__)
        module_init(self)
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert tuple(obs_dict) == ("cam_top", "cam_side", "agent_pos")
        assert obs_dict["cam_top"].shape[1:] == (2, 3, 224, 224)
        assert obs_dict["cam_side"].shape[1:] == (2, 3, 224, 224)
        assert obs_dict["agent_pos"].shape[1:] == (2, 5)
        actions = torch.tensor([[[0.20, 0.0], [0.21, 0.0], [9.0, 9.0]]])
        return {"action": actions.to(self.anchor.device)}


def test_runner_resets_seeds_executes_prefix_and_records_metrics(tmp_path: Path) -> None:
    env = _FakeEnv()
    runner = PaperBaselineRunner(
        tmp_path,
        evaluation_seeds=(100000, 100001),
        n_obs_steps=2,
        n_action_steps=2,
        options={"native_env_factory": lambda: env},
    )
    result = runner.run(_Policy())
    assert env.seeds == [100000, 100001]
    assert np.allclose(env.actions, [[0.2, 0.0], [0.21, 0.0]] * 2)
    assert result["eval/mean_dxy"] == 0.02
    assert result["eval/mean_dyaw"] == 3.0
    assert result["eval/success_rate"] == 1.0
    rollouts = cast("list[dict[str, object]]", result["rollouts"])
    assert rollouts[0] == {
        "seed": 100000,
        "policy_seed": policy_seed(100000),
        "success": True,
        "dxy": 0.02,
        "dyaw": 3.0,
        "duration_s": 0.2,
        "steps": 2,
        "terminated": True,
        "truncated": False,
    }


def test_locked_profiles() -> None:
    assert issubclass(BaseImageDataset, torch.utils.data.Dataset)
    assert tuple(inspect.signature(BaseImageRunner.__init__).parameters) == ("self", "output_dir")
    assert tuple(inspect.signature(BaseImageRunner.run).parameters) == ("self", "policy")
    assert issubclass(BaseImagePolicy, torch.nn.Module)
    assert all(
        issubclass(policy, BaseImagePolicy)
        for policy in (
            DiffusionUnetHybridImagePolicy,
            DiffusionTransformerHybridImagePolicy,
            IbcDfoHybridImagePolicy,
            RobomimicImagePolicy,
        )
    )
    assert TrainRobomimicImageWorkspace.__module__ == (
        "diffusion_policy.workspace.train_robomimic_image_workspace"
    )
    image_normalizer = get_image_range_normalizer()
    assert isinstance(image_normalizer, SingleFieldLinearNormalizer)
    assert image_normalizer.normalize(torch.ones((1, 3, 224, 224))).shape == (1, 3, 224, 224)
    assert isinstance(LinearNormalizer(), LinearNormalizer)
    assert {
        name: (item.observation_steps, item.horizon, item.executed_actions)
        for name, item in PROFILES.items()
    } == {
        "dp_cnn": (2, 16, 8),
        "dp_transformer": (2, 16, 8),
        "ibc": (2, 2, 1),
        "lstm_gmm": (10, 10, 1),
    }
    assert all(
        item.batch_size == 64 and item.optimizer_updates == 100_000 for item in PROFILES.values()
    )
    lstm = workspace_config("lstm_gmm", "/artifact/paper_view/digest", 0)
    assert lstm["_target_"] == (
        "diffusion_policy.workspace.train_robomimic_image_workspace.TrainRobomimicImageWorkspace"
    )
    policy = cast("dict[str, object]", lstm["policy"])
    assert policy["algo_name"] == "bc_rnn"
    training = cast("dict[str, object]", lstm["training"])
    assert training["num_epochs"] == 20
    assert training["max_train_steps"] == 5000
