"""Typed frozen-policy loader for the non-actuating replay path."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from so101_pusht_benchmark.training.artifacts import ArtifactIndex, sha256_file
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity
from so101_pusht_benchmark.training.metadata import read_normalizer_metadata, read_trusted_config


def bind_crop_randomizer_compatibility() -> None:
    """Bind robomimic's moved CropRandomizer class to Stanford's pinned API."""
    base_nets = import_module("robomimic.models.base_nets")
    if hasattr(base_nets, "CropRandomizer"):
        return
    obs_core = import_module("robomimic.models.obs_core")
    crop_randomizer = getattr(obs_core, "CropRandomizer", None)
    if crop_randomizer is None:
        raise RuntimeError("pinned robomimic CropRandomizer implementation is unavailable")
    if crop_randomizer.__module__ != "robomimic.models.obs_core":
        raise RuntimeError("robomimic CropRandomizer provenance drift")
    setattr(base_nets, "CropRandomizer", crop_randomizer)


def load_frozen_policy(
    root: Path, artifact_id: str, model: str
) -> tuple[torch.nn.Module, BundleIdentity]:
    """Load and verify one frozen policy bundle without importing a script."""
    index = ArtifactIndex(root / "artifact-index.json", root)
    record = index.record(artifact_id)
    identity = BundleIdentity.from_dict(record.get("identity"))
    if identity.model != model:
        raise RuntimeError(f"artifact {identity.model} != requested {model}")
    checkpoint = index.verify(artifact_id, "checkpoint")
    config_path = index.verify(artifact_id, "config")
    normalizer = index.verify(artifact_id, "normalizer")
    bundle = index.verify(artifact_id, "bundle")
    checkpoint_digest = sha256_file(checkpoint)
    config_digest = sha256_file(config_path)
    config = read_trusted_config(config_path, identity.model)
    raw_policy = config.get("policy")
    if not isinstance(raw_policy, dict):
        raise TypeError("trusted policy config is not a mapping")
    policy_config = cast("dict[object, object]", raw_policy)
    bind_crop_randomizer_compatibility()
    policy = instantiate(OmegaConf.create(policy_config))
    expected = dict(policy.state_dict())
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    for key, (shape, dtype_name) in read_normalizer_metadata(
        normalizer, identity, checkpoint_digest, config_digest
    ).items():
        expected[key] = torch.empty(shape, dtype=dtypes[dtype_name])
    state = load_bundle(
        bundle,
        expected,
        index=index,
        artifact_id=artifact_id,
        expectation=BundleExpectation(identity, checkpoint_digest),
    )
    policy.load_state_dict(state, strict=True)
    policy.to("cuda:0")
    policy.eval()
    return policy, identity
