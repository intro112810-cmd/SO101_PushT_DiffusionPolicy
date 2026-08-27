"""Four-model tensor-only evaluator in the verified frozen pushT-so100 environment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from collections.abc import Callable
from typing import cast

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from ..evaluation.frozen_env import load_frozen_pusht
from ..integrations.paper_baselines.configs import PROFILES
from ..integrations.paper_baselines.runner import PaperBaselineRunner
from ..native_runtime import NativeRuntimeReport, native_runtime_report
from .artifacts import (
    ArtifactError,
    ArtifactIndex,
    require_production_artifact,
    sha256_file,
)
from .bundle import BundleExpectation, load_bundle
from .identity import BundleIdentity
from .metadata import read_normalizer_metadata, read_trusted_config
from .model_smoke import validate_model_identity

_EVALUATION_SEEDS = tuple(range(100000, 100100))


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    artifact_id: str
    model: str | None = None
    seeds: tuple[int, ...] = _EVALUATION_SEEDS
    device: str = "cuda:0"
    max_steps: int = 300


@dataclass(frozen=True, slots=True)
class EvaluationDependencies:
    environment_factory: Callable[[], object] | None = None
    runtime_report: Callable[[], NativeRuntimeReport] | None = None


def timed_runner_result(
    run: Callable[[], dict[str, object]],
    clock: Callable[[], float],
    synchronize: Callable[[], None],
) -> dict[str, object]:
    """Measure one synchronized evaluation and bind wall time into its metrics."""
    synchronize()
    started = clock()
    result = run()
    synchronize()
    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ArtifactError("evaluation wall time is invalid")
    if "wall_time_s" in result:
        raise ArtifactError("runner result must not provide wall time")
    return {**result, "wall_time_s": elapsed}


def evaluate_bundle(
    bundle: Path,
    output_dir: Path,
    index: ArtifactIndex,
    request: EvaluationRequest,
    dependencies: EvaluationDependencies | None = None,
) -> Path:
    """Strictly reload and atomically evaluate one anchored four-model bundle."""
    selected = EvaluationDependencies() if dependencies is None else dependencies
    selected_runtime_report = (
        native_runtime_report if selected.runtime_report is None else selected.runtime_report
    )
    selected_runtime_report()
    if request.seeds != _EVALUATION_SEEDS:
        raise ArtifactError("evaluation seeds must be exactly ordered 100000..100099")
    if type(request.max_steps) is not int or request.max_steps != 300:
        raise ArtifactError("evaluation step cap must be exactly 300")
    contract = index.authenticate_stage(request.artifact_id, "bundle")
    record = index.record(request.artifact_id)
    require_production_artifact(record, operation="evaluation")
    if (
        contract.get("result_status") != "full_training_bundle_ready"
        or contract.get("identity") != record.get("identity")
        or contract.get("bundle_schema") != 1
    ):
        raise ArtifactError("authenticated bundle producer contract mismatch")
    checkpoint_path = index.verify(request.artifact_id, "checkpoint")
    config_path = index.verify(request.artifact_id, "config")
    normalizer_path = index.verify(request.artifact_id, "normalizer")
    identity = BundleIdentity.from_dict(record.get("identity"))
    if request.model is not None and request.model != identity.model:
        raise ArtifactError("requested model and trusted bundle identity mismatch")
    config = read_trusted_config(config_path, identity.model)
    checkpoint_digest = sha256_file(checkpoint_path)
    config_digest = sha256_file(config_path)
    normalizer_state = read_normalizer_metadata(
        normalizer_path,
        identity,
        checkpoint_digest,
        config_digest,
    )
    policy_raw = cast("dict[str, object]", config["policy"])
    policy_object = instantiate(OmegaConf.create(policy_raw))
    if not isinstance(policy_object, BaseImagePolicy):
        raise ArtifactError("Hydra did not instantiate a pinned upstream policy")
    policy = policy_object
    validate_model_identity(identity.model, policy)
    expected = dict(policy.state_dict())
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    for key, (shape, dtype_name) in normalizer_state.items():
        dtype = dtypes.get(dtype_name)
        if dtype is None:
            raise ArtifactError(f"normalizer metadata dtype is unsupported: {dtype_name}")
        expected[key] = torch.empty(shape, dtype=dtype)
    state = load_bundle(
        bundle,
        expected,
        index=index,
        artifact_id=request.artifact_id,
        expectation=BundleExpectation(identity, checkpoint_digest),
    )
    policy.load_state_dict(state, strict=True)
    device = torch.device(request.device)
    policy.to(device)
    policy.eval()

    profile = PROFILES[identity.model]
    selected_factory = selected.environment_factory or (lambda: load_frozen_pusht(max_steps=300))
    output_dir = output_dir.absolute()
    try:
        output_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ArtifactError(f"output already exists: {output_dir}")
    token = hashlib.sha256(f"evaluation:{request.artifact_id}".encode()).hexdigest()[:12]
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{token}")
    reserved = index.create_output_directory(staging)
    reserved.rmdir()
    published = False
    try:
        runner = PaperBaselineRunner(
            staging,
            evaluation_seeds=request.seeds,
            n_obs_steps=profile.observation_steps,
            n_action_steps=profile.executed_actions,
            options={"max_steps": request.max_steps, "native_env_factory": selected_factory},
        )
        synchronize = (
            (lambda: torch.cuda.synchronize(device))
            if device.type == "cuda"
            else (lambda: None)
        )
        result = timed_runner_result(
            lambda: runner.run(policy),
            time.perf_counter,
            synchronize,
        )
        metrics = {
            "schema": 1,
            "metric_schema": "pusht-so100-dxy-dyaw-v1",
            "model": identity.model,
            "identity": identity.to_dict(),
            "deployment_scope": "simulation_only",
            "training_eligible": index.record(request.artifact_id).get("training_eligible") is True,
            "evaluation_seeds": list(request.seeds),
            "step_cap": request.max_steps,
            "fps": 10,
            "observation_steps": profile.observation_steps,
            "horizon": profile.horizon,
            "executed_actions": profile.executed_actions,
            "optimizer_updates": identity.optimizer_updates,
            **result,
        }
        path = staging / "metrics.json"
        temporary = staging / ".metrics.json.tmp"
        temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        staging.replace(output_dir)
        published = True
        final_path = output_dir / path.name
        index.anchor_evaluation(request.artifact_id, final_path, identity=identity.to_dict())
    except BaseException:
        shutil.rmtree(output_dir if published else staging, ignore_errors=True)
        raise
    else:
        return final_path
