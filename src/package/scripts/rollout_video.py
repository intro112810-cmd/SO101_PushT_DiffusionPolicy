#!/usr/bin/env python3
"""Render one native rollout of a trained bundle to an MP4 (cam_top + cam_side).

Read-only viewer over the verified frozen environment: loads the anchored
bundle via the trusted artifact chain and records one seeded rollout.
Usage:
  python rollout_video.py --artifact dp-cnn-production --seed 100000 [--steps 300]
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
from so101_pusht_benchmark.integrations.paper_baselines.runner import (
    PaperBaselineRunner,
    validate_native_runner_observation,
)
from so101_pusht_benchmark.training.artifacts import ArtifactIndex, sha256_file
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity
from so101_pusht_benchmark.training.metadata import read_normalizer_metadata, read_trusted_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--artifact", default="dp-cnn-production", type=str)
    parser.add_argument("--model", default="dp_cnn", type=str)
    parser.add_argument("--seed", default=100000, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    index = ArtifactIndex(root / "artifact-index.json", root)
    record = index.record(args.artifact)
    identity = BundleIdentity.from_dict(record.get("identity"))
    if args.model != identity.model:
        raise RuntimeError(f"artifact model {identity.model} != requested {args.model}")

    checkpoint_path = index.verify(args.artifact, "checkpoint")
    config_path = index.verify(args.artifact, "config")
    normalizer_path = index.verify(args.artifact, "normalizer")
    bundle_path = index.verify(args.artifact, "bundle")
    checkpoint_digest = sha256_file(checkpoint_path)
    config_digest = sha256_file(config_path)

    config = read_trusted_config(config_path, identity.model)
    normalizer_state = read_normalizer_metadata(
        normalizer_path, identity, checkpoint_digest, config_digest
    )
    policy = instantiate(OmegaConf.create(config["policy"]))
    expected = dict(policy.state_dict())
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    for key, (shape, dtype_name) in normalizer_state.items():
        expected[key] = torch.empty(shape, dtype=dtypes[dtype_name])
    state = load_bundle(
        bundle_path,
        expected,
        index=index,
        artifact_id=args.artifact,
        expectation=BundleExpectation(identity, checkpoint_digest),
    )
    policy.load_state_dict(state, strict=True)
    policy.to("cuda:0")
    policy.eval()

    env = load_frozen_pusht(max_steps=args.steps)
    runner = PaperBaselineRunner(
        root / "tmp-rollout-viewer",
        evaluation_seeds=(),
        n_obs_steps=2,
        n_action_steps=8,
        options={"max_steps": args.steps, "native_env_factory": "frozen"},
    )
    observation, _ = env.reset(seed=args.seed)
    observation = validate_native_runner_observation(observation)
    history: deque[dict[str, np.ndarray]] = deque(
        (observation for _ in range(2)), maxlen=2
    )
    policy.reset()
    frames: list[np.ndarray] = []
    terminated = truncated = False
    with torch.no_grad():
        for _ in range(args.steps):
            prediction = policy.predict_action(runner.policy_observation(history, policy))
            chunk = prediction["action"].detach().cpu().numpy()
            for raw_action in chunk[0]:
                action = np.asarray(raw_action, dtype=np.float32)
                result = env.step(action)
                next_observation = validate_native_runner_observation(result.observation)
                top = np.asarray(next_observation["cam_top"])
                side = np.asarray(next_observation["cam_side"])
                frames.append(np.concatenate([top, side], axis=1))
                history.append(next_observation)
                if result.terminated:
                    terminated = True
                    break
                if result.truncated:
                    truncated = True
                    break
            if terminated or truncated:
                break

    import imageio.v2 as iio_v2

    output = args.output or (root / "reports" / f"rollout_{identity.model}_seed{args.seed}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    iio_v2.mimsave(output, frames, fps=10)
    print(
        f"saved {output} frames={len(frames)} terminated={terminated} truncated={truncated}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
