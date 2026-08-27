from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import so101_pusht_benchmark.data.paper_view as paper_view_module

from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from so101_pusht_benchmark.data.splits import ExperimentConfig, SplitError, freeze_training_view
from so101_pusht_benchmark.integrations.paper_baselines.dataset import (
    DatasetNamespaceError,
    PaperBaselineDataset,
    repeat_training_samples,
    validate_policy_arrays,
    validate_policy_normalizer,
    validate_policy_sample,
)
from so101_pusht_benchmark.workspace import runtime_artifact_root


class _ParameterOwner(Protocol):
    params_dict: torch.nn.ParameterDict


def _source(root: Path, *, held_out_offset: float = 0.0) -> Path:
    count = 10
    length = 3
    rows = count * length
    episode_ids = [f"episode-{index:02d}" for index in range(count)]
    frame = np.tile(np.arange(length, dtype=np.int64), count)
    episode = np.repeat(np.arange(count, dtype=np.int64), length)
    state = np.repeat(episode[:, None], 5, axis=1).astype(np.float32)
    action = np.stack((episode / 20, -episode / 20), axis=1).astype(np.float32)
    state[episode >= 8] += np.float32(held_out_offset)
    action[episode >= 8] += np.float32(held_out_offset)
    arrays = {
        "cam_top": PaperArray(
            np.broadcast_to(episode[:, None, None, None], (rows, 224, 224, 3))
            .astype(np.uint8)
            .copy(),
            "rgb intensity",
        ),
        "cam_side": PaperArray(
            np.broadcast_to((episode + 20)[:, None, None, None], (rows, 224, 224, 3))
            .astype(np.uint8)
            .copy(),
            "rgb intensity",
        ),
        "agent_pos": PaperArray(state, "radians"),
        "action": PaperArray(action, "absolute normalized mocap XY"),
        "timestamp": PaperArray(frame.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(episode, "episode ordinal"),
        "frame_index": PaperArray(frame, "frame ordinal"),
    }
    ends = np.arange(length, rows + 1, length, dtype=np.int64)
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {"fixture.bin": "4" * 64},
        "episodes": [{"episode_id": episode_id, "length": length} for episode_id in episode_ids],
    }
    splits: dict[str, object] = {
        "frozen": False,
        "training_eligible": False,
        "reason": "split_manifest_not_frozen",
        "train": [],
        "validation": [],
        "test": [],
    }
    return write_paper_view(
        root / "source",
        arrays,
        ends,
        PaperViewMetadata(
            canonical_digest(arrays, ends, episode_ids),
            root_provenance_digest(provenance),
            provenance,
            episode_ids,
            splits,
            trusted_runtime_lock_digest(),
            False,
        ),
    )


def _config() -> ExperimentConfig:
    from decimal import Decimal

    return ExperimentConfig(
        schema="pusht-so100-experiment-v1",
        target_episode_count=10,
        split_ratios={
            "train": Decimal("0.8"),
            "validation": Decimal("0.1"),
            "test": Decimal("0.1"),
        },
    )


def test_configurable_budget_freeze_is_deterministic_and_idempotent() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _source(root)
        first, first_manifest = freeze_training_view(source, root / "frozen", _config())
        second, second_manifest = freeze_training_view(source, root / "frozen", _config())
        assert first == second
        assert first_manifest == second_manifest
        assert first_manifest.digest == second_manifest.digest
        assert first_manifest.selected_episode_ids == tuple(
            f"episode-{index:02d}" for index in range(10)
        )


def test_configurable_budget_failure_preserves_readable_source_and_no_output() -> None:
    from decimal import Decimal

    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _source(root)
        insufficient = ExperimentConfig(
            schema="pusht-so100-experiment-v1",
            target_episode_count=50,
            split_ratios={
                "train": Decimal("0.8"),
                "validation": Decimal("0.1"),
                "test": Decimal("0.1"),
            },
        )
        with pytest.raises(SplitError, match=r"accepted episodes 10/50; collect 40 more"):
            freeze_training_view(source, root / "frozen", insufficient)
        assert source.is_dir()
        assert not (root / "frozen").exists()
        assert not list(root.glob(".frozen.tmp-*"))


def test_configurable_budget_cancel_cleans_and_immediately_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _source(root)
        original = paper_view_module.write_array_chunks
        calls = 0

        def interrupt(path: Path, array: NDArray[np.generic], rows: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            original(path, array, rows)

        monkeypatch.setattr(paper_view_module, "write_array_chunks", interrupt)
        with pytest.raises(KeyboardInterrupt):
            freeze_training_view(source, root / "frozen", _config())
        assert source.is_dir()
        assert not (root / "frozen").exists()
        assert not list(root.glob(".frozen.tmp-*"))
        monkeypatch.setattr(paper_view_module, "write_array_chunks", original)
        destination, _ = freeze_training_view(source, root / "frozen", _config())
        assert destination.is_dir()


def test_normalizer_uses_train_only_and_has_both_image_ranges() -> None:
    with (
        TemporaryDirectory(dir=runtime_artifact_root()) as first_tmp,
        TemporaryDirectory(dir=runtime_artifact_root()) as second_tmp,
    ):
        first_root, second_root = Path(first_tmp), Path(second_tmp)
        first, _ = freeze_training_view(_source(first_root), first_root / "frozen", _config())
        second, _ = freeze_training_view(
            _source(second_root, held_out_offset=0.5), second_root / "frozen", _config()
        )
        first_dataset = PaperBaselineDataset(first, horizon=2)
        second_dataset = PaperBaselineDataset(second, horizon=2)
        first_normalizer = first_dataset.get_normalizer()
        second_normalizer = second_dataset.get_normalizer()
        first_stats = first_normalizer.get_input_stats()
        second_stats = second_normalizer.get_input_stats()
        for field in ("action", "agent_pos"):
            for statistic in ("min", "max", "mean", "std"):
                assert torch.equal(first_stats[field][statistic], second_stats[field][statistic])
        assert torch.equal(first_stats["agent_pos"]["max"], torch.full((5,), 7.0))
        for camera in ("cam_top", "cam_side"):
            normalized = first_normalizer[camera].normalize(torch.ones((1, 3, 224, 224)))
            assert normalized.shape == (1, 3, 224, 224)


def test_window_order_is_shared_and_never_crosses_episode_boundaries() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        datasets = [PaperBaselineDataset(frozen, horizon=2) for _model in range(4)]
        expected_windows = datasets[0].ordered_windows
        assert all(dataset.ordered_windows == expected_windows for dataset in datasets)
        assert len(datasets[0]) == 16
        assert expected_windows[:3] == (
            ("episode-00", 0),
            ("episode-00", 1),
            ("episode-01", 0),
        )
        for index in range(len(datasets[0])):
            sample = datasets[0][index]
            observation = cast("dict[str, torch.Tensor]", sample["obs"])
            top = observation["cam_top"][:, 0, 0, 0]
            side = observation["cam_side"][:, 0, 0, 0]
            assert torch.equal(top, top[0].expand_as(top))
            assert torch.equal(side, side[0].expand_as(side))
            assert observation["agent_pos"].shape == (2, 5)
            assert cast("torch.Tensor", sample["action"]).shape == (2, 2)


@pytest.mark.parametrize("horizon", [1, 2, 10, 16])
def test_dual_camera_sample_and_normalizer_contract_for_every_horizon(horizon: int) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        dataset = PaperBaselineDataset(
            frozen,
            horizon=horizon,
            pad_before=min(horizon - 1, 1),
            pad_after=horizon - 1,
        )
        sample = dataset[0]
        observation = cast("dict[str, torch.Tensor]", sample["obs"])
        action = cast("torch.Tensor", sample["action"])
        assert tuple(sample) == ("obs", "action")
        assert tuple(observation) == ("cam_top", "cam_side", "agent_pos")
        assert observation["cam_top"].shape == (horizon, 3, 224, 224)
        assert observation["cam_side"].shape == (horizon, 3, 224, 224)
        assert observation["agent_pos"].shape == (horizon, 5)
        assert action.shape == (horizon, 2)
        assert all(tensor.dtype is torch.float32 for tensor in (*observation.values(), action))
        validate_policy_sample(sample, horizon)
        normalizer = dataset.get_normalizer()
        assert tuple(cast("_ParameterOwner", normalizer).params_dict) == (
            "cam_top",
            "cam_side",
            "agent_pos",
            "action",
        )
        validate_policy_normalizer(normalizer)


def test_dual_camera_namespace_rejects_shape_dtype_keys_and_order() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        dataset = PaperBaselineDataset(frozen, horizon=2)
        view = dataset.view
        cases: list[tuple[str, dict[str, NDArray[np.generic]], str]] = []
        arrays = dict(view.arrays)
        arrays.pop("cam_side")
        cases.append(("single camera", arrays, "array keys/order"))
        arrays = {"cam_side": view.arrays["cam_side"], **view.arrays}
        cases.append(("camera order", arrays, "array keys/order"))
        for key, replacement, match in (
            ("cam_top", np.zeros((30, 224, 224, 3), dtype=np.float32), "cam_top.*uint8"),
            ("cam_side", np.zeros((30, 96, 96, 3), dtype=np.uint8), "cam_side.*224"),
            ("agent_pos", np.zeros((30, 15), dtype=np.float32), "agent_pos.*5"),
            ("action", np.zeros((30, 3), dtype=np.float32), "action.*2"),
        ):
            arrays = dict(view.arrays)
            arrays[key] = replacement
            cases.append((key, arrays, match))
        arrays = dict(view.arrays)
        arrays["unknown"] = np.zeros((30, 1), dtype=np.float32)
        cases.append(("unknown", arrays, "array keys/order"))
        for _name, arrays, match in cases:
            with pytest.raises(DatasetNamespaceError, match=match):
                validate_policy_arrays(arrays, view.episode_ends)


def test_normalizer_and_sample_contract_mismatches_fail_typed() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        dataset = PaperBaselineDataset(frozen, horizon=2)
        normalizer = dataset.get_normalizer()
        del cast("_ParameterOwner", normalizer).params_dict["cam_side"]
        with pytest.raises(DatasetNamespaceError, match="normalizer keys/order"):
            validate_policy_normalizer(normalizer)
        sample = dataset[0]
        observation = cast("dict[str, torch.Tensor]", sample["obs"])
        malformed = {
            "obs": {"cam_side": observation["cam_side"], **observation},
            "action": sample["action"],
        }
        with pytest.raises(DatasetNamespaceError, match="observation keys/order"):
            validate_policy_sample(malformed, 2)
        malformed = {"obs": observation, "action": torch.zeros((2, 3), dtype=torch.float32)}
        with pytest.raises(DatasetNamespaceError, match=r"action.*2"):
            validate_policy_sample(malformed, 2)


def test_window_index_rejects_out_of_range_without_mapping() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        dataset = PaperBaselineDataset(frozen, horizon=2)
        with pytest.raises(IndexError):
            _ = dataset[-1]
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]


def test_full_production_repeat_is_exact_and_validation_remains_finite() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        frozen, _ = freeze_training_view(_source(root), root / "frozen", _config())
        with repeat_training_samples(17):
            dataset = PaperBaselineDataset(frozen, horizon=2)
        assert len(dataset) == 17
        repeated = dataset[16]
        first = dataset[0]
        assert torch.equal(cast("torch.Tensor", repeated["action"]), cast("torch.Tensor", first["action"]))
        validation = dataset.get_validation_dataset()
        assert len(validation) == 2
        with pytest.raises(IndexError):
            _ = validation[2]
