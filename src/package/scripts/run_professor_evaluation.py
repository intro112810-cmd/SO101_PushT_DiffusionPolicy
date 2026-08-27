"""Run one approved model on the fixed 100-seed professor evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from generate_feedback_artifacts import load_policy, observation
from so101_pusht_benchmark.evaluation.frozen_env import (
    FrozenPushTAdapter,
    FrozenStep,
    load_frozen_pusht,
)
from so101_pusht_benchmark.evaluation.professor_artifacts import (
    EVALUATION_SEEDS,
    MODEL_ORDER,
    get_model_spec,
)
from so101_pusht_benchmark.integrations.paper_baselines.runner import (
    PaperBaselineRunner,
)


class PolicyObservationAdapter:
    """Strip private visualization keys before strict policy validation."""

    def __init__(self, environment: FrozenPushTAdapter) -> None:
        self._environment = environment

    def reset(self, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        raw_observation, info = self._environment.reset(seed=seed)
        return dict(observation(raw_observation)), info

    def step(self, action: object) -> FrozenStep:
        result = self._environment.step(action)
        return replace(result, observation=observation(result.observation))

    def close(self) -> None:
        self._environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_ORDER)
    parser.add_argument("--artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = get_model_spec(args.model)
    policy, identity = load_policy(root, args.artifact or spec.artifact_id, spec.model)
    runner = PaperBaselineRunner(
        output,
        evaluation_seeds=EVALUATION_SEEDS,
        n_obs_steps=identity.observation_steps,
        n_action_steps=identity.executed_actions,
        options={
            "max_steps": 300,
            "native_env_factory": lambda: PolicyObservationAdapter(
                load_frozen_pusht(max_steps=300)
            ),
        },
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = runner.run(policy)
    torch.cuda.synchronize()
    metrics = {
        "schema": 1,
        "metric_schema": "pusht-so100-dxy-dyaw-v1",
        "model": identity.model,
        "identity": identity.to_dict(),
        "deployment_scope": "simulation_only",
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "step_cap": 300,
        "fps": 10,
        "observation_steps": identity.observation_steps,
        "horizon": identity.horizon,
        "executed_actions": identity.executed_actions,
        "optimizer_updates": identity.optimizer_updates,
        "wall_time_s": time.perf_counter() - started,
        **result,
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"published {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
