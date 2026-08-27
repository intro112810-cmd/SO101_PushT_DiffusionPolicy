"""Model-agnostic launcher for the unchanged upstream training workspaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

from omegaconf import OmegaConf

from ..integrations.paper_baselines.configs import workspace_config
from ..integrations.paper_baselines.dataset import repeat_training_samples
from .model_smoke import (
    bounded_policy_config,
    resolve_workspace_class,
    run_workspace_one_batch,
    validate_profile_config,
    validate_profile_origin,
    validate_smoke_store,
)
from .artifacts import ArtifactError, ArtifactIndex, ArtifactScope
from .identity import fixture_split_digest, trusted_identity
from .runtime import assert_paper_runtime


@dataclass(frozen=True, slots=True)
class TrainingLaunch:
    seed: int
    artifact_id: str
    simulation_probe: bool = False
    model: str = "dp_cnn"
    smoke: bool = False
    smoke_mode: Literal["fixture", "production"] | None = None
    training_mode: Literal["full_production"] | None = None
    max_updates: int | None = None


MODEL_NAMES = ("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")


def _run_with_periodic_checkpoints(workspace: _Workspace, config: dict[str, object]) -> None:
    """Run the workspace while checkpointing every N steps (local exploratory mode).

    The upstream workspaces only checkpoint at epoch boundaries, but local
    full-production runs execute the whole budget inside one epoch (via
    repeat_training_samples). This helper watches global_step from a separate
    thread and calls save_checkpoint() (tag='latest') every interval so an
    interrupted run keeps its latest state. Interval comes from the
    PUSHT_CHECKPOINT_EVERY env var (default 25_000 steps).
    """
    interval = 25_000
    try:
        raw = int(os.environ.get("PUSHT_CHECKPOINT_EVERY", "25000"))
        if raw > 0:
            interval = raw
    except ValueError:
        pass
    stop = threading.Event()

    def _watcher() -> None:
        last = 0
        while not stop.is_set():
            current = getattr(workspace, "global_step", 0)
            if current - last >= interval and current > 0:
                try:
                    workspace.save_checkpoint()
                    last = current
                except BaseException:
                    # Never kill training because a checkpoint write failed.
                    pass
            stop.wait(0.2)

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()
    try:
        workspace.run()
    finally:
        stop.set()
        watcher.join(timeout=60.0)


class _Workspace(Protocol):
    __dict__: dict[str, object]
    global_step: int

    def run(self) -> None: ...

    def save_checkpoint(self, *args: object, **kwargs: object) -> str | Path: ...

    def load_checkpoint(self, *args: object, **kwargs: object) -> object: ...


class _CheckpointWriterOwner:
    """Retain every upstream checkpoint writer until the transaction is closed."""

    def __init__(self, workspace: _Workspace) -> None:
        self._workspace = workspace
        self._save = workspace.save_checkpoint
        self._writers: list[threading.Thread] = []
        self._subscribed_save = self._save_and_capture

    def _capture(self) -> None:
        writer = self._workspace.__dict__.get("_saving_thread")
        if writer is not None and not isinstance(writer, threading.Thread):
            raise TypeError("upstream workspace saving thread is invalid")
        if writer is not None and all(writer is not owned for owned in self._writers):
            self._writers.append(writer)

    def _save_and_capture(self, *args: object, **kwargs: object) -> str | Path:
        result = self._save(*args, **kwargs)
        self._capture()
        return result

    def subscribe(self) -> None:
        self._workspace.__dict__["save_checkpoint"] = self._subscribed_save

    def close(self) -> None:
        if self._workspace.__dict__.get("save_checkpoint") is self._subscribed_save:
            del self._workspace.__dict__["save_checkpoint"]
        self._capture()
        for writer in self._writers:
            writer.join()

    def save_final_checkpoint(self) -> str | Path:
        result = self._save(use_thread=False)
        self._capture()
        for writer in self._writers:
            writer.join()
        return result


def resolved_config(model: str, paper_view: Path, seed: int) -> dict[str, object]:
    """Resolve the approved profile for a model without changing upstream targets."""
    return workspace_config(model, paper_view, seed)


def resolved_dp_cnn_config(paper_view: Path, seed: int) -> dict[str, object]:
    """Resolve the one approved DP-CNN profile (backward-compatible alias)."""
    return resolved_config("dp_cnn", paper_view, seed)


def update_budget(config: dict[str, object]) -> int:
    training = config.get("training")
    if not isinstance(training, dict):
        raise TypeError("workspace training config is missing")
    typed = cast("dict[str, object]", training)
    steps, epochs = typed.get("max_train_steps"), typed.get("num_epochs")
    if not isinstance(steps, int) or not isinstance(epochs, int):
        raise TypeError("workspace update budget is invalid")
    return steps * epochs


def synthetic_probe_config(config: dict[str, object]) -> dict[str, object]:
    """Return a bounded, ineligible profile using the original policy and workspace classes."""
    probe = deepcopy(config)
    training = cast("dict[str, object]", probe["training"])
    training.update(
        {
            "debug": False,
            "resume": False,
            "num_epochs": 3,
            "max_train_steps": 10,
            "max_val_steps": 2,
            "lr_warmup_steps": 0,
        }
    )
    name = str(probe["name"])
    policy = cast("dict[str, object]", probe["policy"])
    if name == "dp_cnn":
        policy.update({"down_dims": [32, 64, 128], "diffusion_step_embed_dim": 32})
    elif name == "dp_transformer":
        policy.update({"n_layer": 1, "n_head": 4, "n_emb": 32, "n_cond_layers": 0})
    elif name == "ibc":
        policy.update({"train_n_neg": 2, "pred_n_iter": 1, "pred_n_samples": 2})
    optimizer = probe.get("optimizer")
    if isinstance(optimizer, dict) and "lr" in optimizer:
        cast("dict[str, object]", optimizer)["lr"] = 0.001
    dataloader = cast("dict[str, object]", probe["dataloader"])
    validation = cast("dict[str, object]", probe["val_dataloader"])
    dataloader.update({"batch_size": 2, "num_workers": 0})
    validation.update({"batch_size": 2, "num_workers": 0})
    task = cast("dict[str, object]", probe["task"])
    dataset = cast("dict[str, object]", task["dataset"])
    dataset["split"] = "synthetic_probe"
    runner = cast("dict[str, object]", task["env_runner"])
    runner.update({"evaluation_seeds": [100000], "options": {"max_steps": 1}})
    logging = cast("dict[str, object]", probe["logging"])
    logging["name"] = f"{name}_synthetic_probe"
    return probe


def full_production_config(
    config: dict[str, object], updates: int = 100_000
) -> dict[str, object]:
    """Execute the approved updates while deferring rollouts to final evaluation."""
    production = deepcopy(config)
    training = cast("dict[str, object]", production["training"])
    if updates <= 0:
        raise ArtifactError("approved update budget must be positive")
    training["num_epochs"] = 1
    training["max_train_steps"] = updates
    training["resume"] = False
    training["rollout_every"] = 2
    task = cast("dict[str, object]", production["task"])
    runner = cast("dict[str, object]", task["env_runner"])
    options = cast("dict[str, object]", runner.setdefault("options", {}))
    options["native_env_factory"] = "frozen"
    runner["evaluation_seeds"] = []
    return production


def full_production_sample_count(config: dict[str, object]) -> int:
    """Return the exact number of train samples needed for the locked update budget."""
    dataloader = cast("dict[str, object]", config["dataloader"])
    batch_size = dataloader.get("batch_size")
    if type(batch_size) is not int or batch_size < 1:
        raise ArtifactError("full production batch size is invalid")
    return update_budget(config) * batch_size


def remaining_full_production_sample_count(
    config: dict[str, object], completed_updates: int
) -> int:
    """Return samples needed to finish, rather than exceed, the locked budget."""
    total_updates = update_budget(config)
    if type(completed_updates) is not int or completed_updates < 0:
        raise ArtifactError("resume checkpoint update count is invalid")
    if completed_updates >= total_updates:
        raise ArtifactError("resume checkpoint already reached the locked budget")
    dataloader = cast("dict[str, object]", config["dataloader"])
    batch_size = dataloader.get("batch_size")
    if type(batch_size) is not int or batch_size < 1:
        raise ArtifactError("full production batch size is invalid")
    return (total_updates - completed_updates) * batch_size


def restore_workspace_output_dir(workspace: _Workspace, output_dir: Path) -> None:
    """Keep a resumed workspace's generated files in its current staging directory."""
    workspace.__dict__["_output_dir"] = str(output_dir)


def smoke_probe_config(
    config: dict[str, object], *, mode: Literal["fixture", "production"] = "fixture"
) -> dict[str, object]:
    """Return a one-update smoke profile using the original classes."""
    probe = deepcopy(config)
    training = cast("dict[str, object]", probe["training"])
    training.update(
        {
            "debug": False,
            "resume": False,
            "num_epochs": 1,
            "max_train_steps": 1,
            "max_val_steps": 2,
            "lr_warmup_steps": 0,
            "rollout_every": 2,
            "checkpoint_every": 2,
            "val_every": 2,
            "sample_every": 2,
        }
    )
    probe["policy"] = bounded_policy_config(str(probe["name"]))
    dataloader = cast("dict[str, object]", probe["dataloader"])
    validation = cast("dict[str, object]", probe["val_dataloader"])
    dataloader.update({"batch_size": 1, "num_workers": 0})
    validation.update({"batch_size": 1, "num_workers": 0})
    task = cast("dict[str, object]", probe["task"])
    dataset = cast("dict[str, object]", task["dataset"])
    if mode == "fixture":
        dataset["split"] = "synthetic_probe"
    elif mode != "production":
        raise ValueError(f"unknown smoke mode: {mode}")
    return probe


def preflight_full_production(
    paper_view: Path,
    model: str,
    seed: int,
    max_updates: int,
) -> dict[str, object]:
    """Validate the complete full-training contract without creating output or training."""
    assert_paper_runtime()
    if model not in MODEL_NAMES:
        raise ArtifactError(f"unknown model: {model}")
    config = resolved_config(model, paper_view, seed)
    validate_profile_config(model, config)
    training = cast("dict[str, object]", config["training"])
    training["max_train_steps"] = max_updates
    training["num_epochs"] = 1
    config = full_production_config(config, max_updates)
    configured_updates = update_budget(config)
    if max_updates != configured_updates or configured_updates <= 0:
        raise ArtifactError("full production requires the approved positive update budget")
    try:
        store_identity = validate_smoke_store(paper_view, mode="production")
    except ValueError as exc:
        raise ArtifactError(f"full production requires immutable eligible input: {exc}") from exc
    if store_identity.split_digest is None:
        raise ArtifactError("full production split digest is missing")
    validate_profile_origin(model)
    identity = trusted_identity(
        model,
        store_identity.canonical_digest,
        store_identity.split_digest,
        optimizer_updates=configured_updates,
    )
    return {
        "status": "full-production-preflight",
        "model": model,
        "training_mode": "full_production",
        "configured_optimizer_updates": configured_updates,
        "rollout_during_training": False,
        "identity": identity.to_dict(),
        "artifacts_created": False,
    }


def launch_training(
    paper_view: Path,
    output_dir: Path,
    index: ArtifactIndex,
    request: TrainingLaunch,
) -> Path:
    """Run the profile's workspace lifecycle and anchor its native checkpoint."""
    assert_paper_runtime()
    configure_cuda_runtime()
    if request.model not in MODEL_NAMES:
        raise ArtifactError(f"unknown model: {request.model}")
    config = resolved_config(request.model, paper_view, request.seed)
    validate_profile_config(request.model, config)
    if request.training_mode == "full_production" and request.max_updates is not None:
        training = cast("dict[str, object]", config["training"])
        training["max_train_steps"] = request.max_updates
        training["num_epochs"] = 1
    configured_updates = update_budget(config)
    if configured_updates <= 0:
        raise ArtifactError("update budget is missing or invalid")
    smoke_mode = request.smoke_mode or ("fixture" if request.smoke else None)
    full_production = request.training_mode == "full_production"
    if request.training_mode not in {None, "full_production"}:
        raise ArtifactError("unknown training mode")
    if request.simulation_probe and (smoke_mode is not None or full_production):
        raise ArtifactError("simulation probe, smoke, and full production are mutually exclusive")
    if smoke_mode is not None and full_production:
        raise ArtifactError("smoke and full production are mutually exclusive")
    if request.max_updates is not None and not full_production:
        raise ArtifactError("max_updates is valid only for full production")
    if not request.simulation_probe and smoke_mode is None and not full_production:
        raise ArtifactError("training mode must be smoke or full_production")
    if full_production and request.max_updates != configured_updates:
        raise ArtifactError(
            "full production requires explicit max_updates matching the approved profile"
        )
    if request.simulation_probe:
        config = synthetic_probe_config(config)
    elif full_production:
        config = full_production_config(config, configured_updates)
        task = cast("dict[str, object]", config["task"])
        runner = cast("dict[str, object]", task["env_runner"])
        runner_seeds = runner.get("evaluation_seeds")
        if runner_seeds:
            raise ArtifactError(
                f"full production must defer rollouts, got evaluation_seeds={runner_seeds}"
            )
    store_identity = None
    if smoke_mode is not None or full_production:
        selected_store_mode: Literal["fixture", "production"] = (
            "production"
            if full_production
            else cast("Literal['fixture', 'production']", smoke_mode)
        )
        try:
            store_identity = validate_smoke_store(paper_view, mode=selected_store_mode)
        except ValueError as exc:
            if full_production:
                raise ArtifactError(
                    f"full production requires immutable eligible input: {exc}"
                ) from exc
            raise ArtifactError(str(exc)) from exc
    if smoke_mode is not None:
        config = smoke_probe_config(config, mode=smoke_mode)
    validate_profile_origin(request.model)
    workspace_cls = resolve_workspace_class(request.model)
    if str(config["_target_"]) != f"{workspace_cls.__module__}.{workspace_cls.__name__}":
        raise ArtifactError("workspace target does not match locked upstream origin")
    output_dir = output_dir.absolute()
    try:
        output_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ArtifactError(f"output already exists: {output_dir}")
    stage_token = hashlib.sha256(request.artifact_id.encode()).hexdigest()[:12]
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{stage_token}")
    staging = index.create_output_directory(
        staging, allow_existing=os.environ.get("PUSHT_LOCAL_RESUME") == "1"
    )
    published = False
    checkpoint: Path
    identity: dict[str, object] | None = None
    if full_production:
        assert store_identity is not None
        assert store_identity.split_digest is not None
        identity = trusted_identity(
            request.model,
            store_identity.canonical_digest,
            store_identity.split_digest,
            optimizer_updates=configured_updates,
        ).to_dict()
    try:
        config_path = staging / "resolved_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        workspace_factory = cast("Callable[..., _Workspace]", workspace_cls)
        workspace = workspace_factory(OmegaConf.create(config), output_dir=str(staging))
        training_sample_count = full_production_sample_count(config)
        if os.environ.get("PUSHT_LOCAL_RESUME") == "1":
            checkpoint = staging / "checkpoints" / "latest.ckpt"
            if checkpoint.is_file():
                workspace.load_checkpoint(path=str(checkpoint))
                restore_workspace_output_dir(workspace, staging)
                print(
                    f"resumed from checkpoint at global_step={workspace.global_step}",
                    flush=True,
                )
                training_sample_count = remaining_full_production_sample_count(
                    config, workspace.global_step
                )
            else:
                print("PUSHT_LOCAL_RESUME=1 but no checkpoint found — starting fresh", flush=True)
        writers = _CheckpointWriterOwner(workspace)
        writers.subscribe()
        try:
            try:
                if smoke_mode is None:
                    if full_production:
                        with repeat_training_samples(training_sample_count):
                            if os.environ.get("PUSHT_LOCAL_BUDGET") == "1":
                                _run_with_periodic_checkpoints(workspace, config)
                            else:
                                workspace.run()
                    else:
                        workspace.run()
                    if full_production:
                        executed_updates = workspace.global_step
                        if (
                            type(executed_updates) is not int
                            or executed_updates != configured_updates
                        ):
                            raise ArtifactError(
                                "executed optimizer updates do not match the locked budget"
                            )
                        (staging / "training_receipt.json").write_text(
                            json.dumps(
                                {
                                    "schema": "pusht-so100-full-training-v1",
                                    "model": request.model,
                                    "training_mode": "full_production",
                                    "configured_optimizer_updates": configured_updates,
                                    "executed_optimizer_updates": executed_updates,
                                    "rollout_during_training": False,
                                    "completed": True,
                                    "identity": identity,
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                else:
                    receipt = run_workspace_one_batch(
                        workspace, request.model, paper_view, mode=smoke_mode
                    )
                    dataset_value = receipt["canonical_digest"]
                    if not isinstance(dataset_value, str):
                        raise ArtifactError("smoke receipt canonical digest is invalid")
                    dataset_digest = dataset_value
                    split_value = receipt["split_digest"]
                    split_digest = (
                        split_value
                        if isinstance(split_value, str)
                        else fixture_split_digest(dataset_digest)
                    )
                    identity = trusted_identity(
                        request.model, dataset_digest, split_digest
                    ).to_dict()
                    (staging / "smoke_receipt.json").write_text(
                        json.dumps(
                            {**receipt, "trusted_identity": identity}, indent=2, sort_keys=True
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            finally:
                finish_value = __import__("wandb").__dict__.get("finish")
                if not callable(finish_value):
                    raise TypeError("wandb runtime has no finish lifecycle hook")
                cast("Callable[[], None]", finish_value)()
        finally:
            writers.close()
        staged_checkpoint = Path(writers.save_final_checkpoint()).resolve()
        checkpoint_relative = staged_checkpoint.relative_to(staging)
        staging.replace(output_dir)
        published = True
        checkpoint = output_dir / checkpoint_relative
        config_path = output_dir / "resolved_config.json"
        production_receipt = output_dir / "training_receipt.json" if full_production else None
        index.anchor_checkpoint(
            request.artifact_id,
            checkpoint,
            ArtifactScope(
                config=config_path,
                simulation_probe=request.simulation_probe,
                smoke_mode=smoke_mode,
                training_mode=request.training_mode,
                identity=identity,
                production_receipt=production_receipt,
                training_log=output_dir / "logs.json.txt" if full_production else None,
            ),
        )
    except BaseException:
        target = output_dir if published else staging
        shutil.rmtree(target, ignore_errors=True)
        raise
    return checkpoint


def configure_cuda_runtime() -> None:
    """Apply scoped CUDA workarounds requested by the training launcher."""
    if os.environ.get("PUSHT_DISABLE_CUDNN") != "1":
        return
    import torch

    torch.backends.cudnn.enabled = False
