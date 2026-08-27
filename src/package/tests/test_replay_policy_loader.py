from __future__ import annotations

from importlib import import_module

from so101_pusht_benchmark.sim_to_real.replay_policy_loader import (
    bind_crop_randomizer_compatibility,
)


def test_crop_randomizer_compatibility_binds_the_pinned_moved_class() -> None:
    base_nets = import_module("robomimic.models.base_nets")
    obs_core = import_module("robomimic.models.obs_core")

    bind_crop_randomizer_compatibility()

    assert base_nets.CropRandomizer is obs_core.CropRandomizer
