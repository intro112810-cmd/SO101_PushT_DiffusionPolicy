from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml


@dataclass(frozen=True, slots=True)
class PaperModelProfile:
    budget: dict[str, object]
    resolved_optimizer_updates: int
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class PaperProfiles:
    training_seeds: tuple[int, ...]
    models: MappingProxyType[str, PaperModelProfile]


def load_paper_profiles(path: Path) -> PaperProfiles:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("paper profile config must be an object")
    value = cast("dict[str, object]", raw)
    if value.get("schema") != "pusht-so100-paper-faithful-profiles-v1":
        raise ValueError("paper profile schema mismatch")
    seeds_raw: object = value.get("training_seeds")
    models_raw = value.get("models")
    if (
        not isinstance(seeds_raw, list)
        or seeds_raw != [0, 1, 2]
        or not all(type(seed) is int for seed in cast("list[object]", seeds_raw))
        or not isinstance(models_raw, dict)
    ):
        raise ValueError("paper profile seeds or models are malformed")
    models: dict[str, PaperModelProfile] = {}
    for name, raw_profile in cast("dict[str, object]", models_raw).items():
        if not isinstance(raw_profile, dict):
            raise TypeError(f"model profile is malformed: {name}")
        profile = cast("dict[str, object]", raw_profile)
        budget = profile.get("budget")
        updates = profile.get("resolved_optimizer_updates")
        parameters = profile.get("parameters")
        if (
            not isinstance(budget, dict)
            or type(updates) is not int
            or cast(int, updates) <= 0
            or not isinstance(parameters, dict)
        ):
            raise TypeError(f"model profile fields are malformed: {name}")
        models[name] = PaperModelProfile(
            cast("dict[str, object]", budget),
            cast(int, updates),
            cast("dict[str, object]", parameters),
        )
    if set(models) != {"dp_cnn", "dp_transformer", "ibc", "lstm_gmm"}:
        raise ValueError("paper profile model set mismatch")
    return PaperProfiles(tuple(cast("list[int]", seeds_raw)), MappingProxyType(models))
