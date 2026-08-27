from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from so101_pusht_benchmark.data.splits import (
    ExperimentConfig,
    SplitError,
    allocate_split_counts,
    build_split_manifest,
    load_experiment_config,
    validate_split_manifest,
)


CONFIG = Path(__file__).parents[1] / "configs/experiment/pusht_so100_four_model_v1.yaml"


def _ids(count: int) -> list[str]:
    return [f"episode-{index:03d}" for index in range(count)]


def _config(target: int, ratios: tuple[str, str, str] = ("0.8", "0.1", "0.1")) -> ExperimentConfig:
    return ExperimentConfig(
        schema="pusht-so100-experiment-v1",
        target_episode_count=target,
        split_ratios={
            "train": Decimal(ratios[0]),
            "validation": Decimal(ratios[1]),
            "test": Decimal(ratios[2]),
        },
    )


def test_configurable_budget_default_config_is_versioned_and_exact() -> None:
    config = load_experiment_config(CONFIG)
    assert config.schema == "pusht-so100-experiment-v1"
    assert config.target_episode_count == 200
    assert config.split_ratios == {
        "train": Decimal("0.8"),
        "validation": Decimal("0.1"),
        "test": Decimal("0.1"),
    }


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (10, {"train": 8, "validation": 1, "test": 1}),
        (50, {"train": 40, "validation": 5, "test": 5}),
        (200, {"train": 160, "validation": 20, "test": 20}),
    ],
)
def test_configurable_budget_deterministic_manifests(target: int, expected: dict[str, int]) -> None:
    eligible = list(reversed(_ids(target + 7)))
    config = _config(target)
    first = build_split_manifest(eligible, config, source_digest="a" * 64)
    second = build_split_manifest(eligible, config, source_digest="a" * 64)
    assert first == second
    assert first.digest == second.digest
    assert first.selected_episode_ids == tuple(_ids(target))
    assert {name: len(first.members(name)) for name in expected} == expected
    assert sum(expected.values()) == len(first.selected_episode_ids) == target
    assert all(first.members(name) for name in expected)
    validate_split_manifest(first, eligible, config, source_digest="a" * 64)


def test_split_largest_remainder_has_declared_tie_order() -> None:
    counts = allocate_split_counts(_config(7, ("0.5", "0.3", "0.2")))
    assert counts == {"train": 4, "validation": 2, "test": 1}


def test_split_sessions_are_disjoint_or_fail_closed() -> None:
    eligible = _ids(50)
    sessions = {
        **dict.fromkeys(eligible[:40], "session-train"),
        **dict.fromkeys(eligible[40:45], "session-validation"),
        **dict.fromkeys(eligible[45:], "session-test"),
    }
    manifest = build_split_manifest(
        eligible, _config(50), source_digest="b" * 64, sessions=sessions
    )
    session_sets = [
        {sessions[item] for item in manifest.members(name)}
        for name in ("train", "validation", "test")
    ]
    assert session_sets[0].isdisjoint(session_sets[1])
    assert session_sets[0].isdisjoint(session_sets[2])
    assert session_sets[1].isdisjoint(session_sets[2])

    impossible = {
        **dict.fromkeys(_ids(4), "session-a"),
        **dict.fromkeys(_ids(10)[4:7], "session-b"),
        **dict.fromkeys(_ids(10)[7:], "session-c"),
    }
    with pytest.raises(SplitError, match="session-disjoint allocation impossible"):
        build_split_manifest(_ids(10), _config(10), source_digest="c" * 64, sessions=impossible)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (_config(2), "target_episode_count must be at least 3"),
        (_config(10, ("0", "0", "1")), "train and validation ratios must be positive"),
        (_config(10, ("-0.1", "0.6", "0.5")), "train and validation ratios must be positive"),
        (_config(10, ("0.7", "0.2", "0.2")), "split ratios must sum to 1"),
        (_config(3, ("0.98", "0.01", "0.01")), "non-empty positive-ratio allocation impossible"),
    ],
)
def test_configurable_budget_rejects_invalid_config(config: ExperimentConfig, error: str) -> None:
    with pytest.raises(SplitError, match=error):
        allocate_split_counts(config)


def test_configurable_budget_allows_no_offline_test_partition() -> None:
    config = _config(200, ("0.9", "0.1", "0"))

    assert allocate_split_counts(config) == {"train": 180, "validation": 20, "test": 0}


def test_configurable_budget_reports_exact_progress_and_rejects_duplicates() -> None:
    with pytest.raises(SplitError, match=r"accepted episodes 49/50; collect 1 more"):
        build_split_manifest(_ids(49), _config(50), source_digest="d" * 64)
    duplicated = [*_ids(10), "episode-009"]
    with pytest.raises(SplitError, match="duplicate eligible episode ID"):
        build_split_manifest(duplicated, _config(10), source_digest="d" * 64)


def test_split_manifest_rejects_overlap_and_digest_tamper() -> None:
    eligible = _ids(10)
    manifest = build_split_manifest(eligible, _config(10), source_digest="e" * 64)
    overlapping = replace(
        manifest,
        validation=(manifest.train[0],),
    )
    with pytest.raises(SplitError, match=r"overlap|partition"):
        validate_split_manifest(overlapping, eligible, _config(10), source_digest="e" * 64)
    tampered = replace(manifest, digest="f" * 64)
    with pytest.raises(SplitError, match="digest mismatch"):
        validate_split_manifest(tampered, eligible, _config(10), source_digest="e" * 64)


def test_split_session_metadata_must_be_complete() -> None:
    sessions = {item: f"session-{index}" for index, item in enumerate(_ids(9))}
    with pytest.raises(SplitError, match="session metadata must cover every selected episode"):
        build_split_manifest(_ids(10), _config(10), source_digest="f" * 64, sessions=sessions)
