"""Bounded real-upstream model training proofs for fixture and production smoke modes."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from collections.abc import Callable
from typing import Literal, Protocol, cast

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from ..data.paper_view_reader import load_paper_view, validate_training_view
from ..integrations.paper_baselines.configs import (
    PROFILES,
    SHAPE_META,
    observation_encoder,
    policy_config,
    validate_shape_meta,
)
from ..integrations.paper_baselines.dataset import PaperBaselineDataset
from ..integrations.paper_baselines.robomimic import assert_lstm_gmm_runtime

SmokeMode = Literal["fixture", "production"]
_STANFORD_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
_ROBOMIMIC_COMMIT = "62ed2de905caeb9133136e4d14d810a8b6baa96c"


@dataclass(frozen=True, slots=True)
class SmokeStoreIdentity:
    mode: SmokeMode
    canonical_digest: str
    split_digest: str | None
    training_eligible: bool
    comparison_eligible: bool


@dataclass(frozen=True, slots=True)
class _UpdateMetrics:
    loss: float
    gradient_squared_norm: float
    parameter_delta: float


class _RobomimicTrainPolicy(Protocol):
    def train_on_batch(
        self, batch: dict[str, object], epoch: int, validate: bool = False
    ) -> dict[str, object]: ...


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _require_pinned_source(symbol: type[object], expected: Path, label: str) -> Path:
    source = Path(inspect.getfile(symbol)).resolve()
    if not expected.is_file() or source != expected.resolve():
        raise TypeError(f"{label} origin is not the pinned upstream checkout: {source}")
    return source


def resolve_workspace_class(model: str) -> type[object]:
    """Resolve exactly one locked Stanford workspace without dynamic-origin fallback."""
    profile = PROFILES.get(model)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {model}")
    module_name, class_name = profile.workspace_target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    workspace = getattr(module, class_name, None)
    if not isinstance(workspace, type):
        raise TypeError(f"locked workspace class is unavailable: {profile.workspace_target}")
    expected = (
        _project_root()
        / "04_experiments/so101_pusht_benchmark/cache/upstream/stanford"
        / Path(*module_name.split("."))
    ).with_suffix(".py")
    _require_pinned_source(workspace, expected, "workspace")
    return workspace


def validate_profile_config(model: str, config: object) -> None:
    """Reject model selection, shape, horizon, class, or workspace identity drift."""
    profile = PROFILES.get(model)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {model}")
    if not isinstance(config, dict):
        raise TypeError("workspace profile must be a mapping")
    values = cast("dict[str, object]", config)
    expected = {
        "_target_": profile.workspace_target,
        "name": model,
        "horizon": profile.horizon,
        "n_obs_steps": profile.observation_steps,
        "n_action_steps": profile.executed_actions,
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise TypeError(f"{model} profile {key} mismatch")
    validate_shape_meta(values.get("shape_meta"))
    policy = values.get("policy")
    if not isinstance(policy, dict):
        raise TypeError(f"{model} policy profile is missing")
    policy_values = cast("dict[str, object]", policy)
    expected_policy = f"{profile.policy_class.__module__}.{profile.policy_class.__name__}"
    if policy_values.get("_target_") != expected_policy:
        raise TypeError(f"{model} policy target mismatch")
    validate_shape_meta(policy_values.get("shape_meta"))


def validate_profile_origin(model: str) -> tuple[Path, Path]:
    """Validate selected policy and workspace definitions before creating output."""
    profile = PROFILES.get(model)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {model}")
    policy_module = profile.policy_class.__module__
    expected_policy = (
        _project_root()
        / "04_experiments/so101_pusht_benchmark/cache/upstream/stanford"
        / Path(*policy_module.split("."))
    ).with_suffix(".py")
    policy_origin = _require_pinned_source(profile.policy_class, expected_policy, "policy")
    workspace = resolve_workspace_class(model)
    return policy_origin, Path(inspect.getfile(workspace)).resolve()


def validate_model_identity(model: str, policy: object) -> tuple[Path, Path]:
    """Fail closed unless policy, workspace, and nested encoder are pinned real classes."""
    profile = PROFILES.get(model)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {model}")
    validate_shape_meta(SHAPE_META)
    if type(policy) is not profile.policy_class:
        raise TypeError(f"{model} policy class is not the locked upstream class")
    policy_origin, workspace_origin = validate_profile_origin(model)
    observation_encoder(policy)
    if model == "lstm_gmm":
        assert_lstm_gmm_runtime(policy)
    return policy_origin, workspace_origin


def validate_smoke_store(store: str | Path, *, mode: SmokeMode) -> SmokeStoreIdentity:
    """Separate synthetic fixture proof from immutable production smoke eligibility."""
    path = Path(store)
    if mode == "fixture":
        view = load_paper_view(path)
        splits = view.splits
        provenance = view.manifest.get("root_provenance")
        provenance_values = (
            cast("dict[str, object]", provenance) if isinstance(provenance, dict) else {}
        )
        members_raw = provenance_values.get("source_members")
        members = cast("dict[str, object]", members_raw) if isinstance(members_raw, dict) else {}
        if (
            view.manifest.get("training_eligible") is not False
            or splits.get("training_eligible") is not False
            or splits.get("frozen") is not False
            or splits.get("reason") != "synthetic_fixture_not_comparison_eligible"
            or not members
            or not any(name.startswith("synthetic-fixture") for name in members)
        ):
            raise ValueError("fixture smoke requires a clearly ineligible synthetic store")
        canonical_digest = view.manifest.get("canonical_digest")
        if not isinstance(canonical_digest, str):
            raise ValueError("fixture smoke store canonical digest is missing")
        return SmokeStoreIdentity("fixture", canonical_digest, None, False, False)
    if mode != "production":
        raise ValueError(f"unknown smoke mode: {mode}")
    try:
        view = validate_training_view(path)
    except Exception as exc:
        raise ValueError("production smoke requires immutable frozen manifest and digest") from exc
    split_digest = view.splits.get("digest")
    if not isinstance(split_digest, str) or len(split_digest) != 64:
        raise ValueError("production smoke requires immutable frozen manifest and digest")
    # Smoke output is never a final comparison result; Todo 9 must apply later gates.
    canonical_digest = view.manifest.get("canonical_digest")
    if not isinstance(canonical_digest, str):
        raise TypeError("production smoke requires immutable frozen manifest and digest")
    return SmokeStoreIdentity("production", canonical_digest, split_digest, True, False)


def bounded_policy_config(model: str) -> dict[str, object]:
    config = policy_config(model)
    config["crop_shape"] = [32, 32]
    if model == "dp_cnn":
        config.update(diffusion_step_embed_dim=32, down_dims=[32, 64])
    elif model == "dp_transformer":
        config.update(n_layer=1, n_head=4, n_emb=32, n_cond_layers=0)
    elif model == "ibc":
        config.update(train_n_neg=2, pred_n_iter=1, pred_n_samples=2)
    return config


def _batch(store: str | Path, model: str, split: str) -> dict[str, object]:
    profile = PROFILES[model]
    dataset = PaperBaselineDataset(
        store,
        horizon=profile.horizon,
        pad_before=0,
        pad_after=0,
        split=split,
    )
    sample = dataset[0]
    observations = cast("dict[str, torch.Tensor]", sample["obs"])
    return {
        "obs": {key: value.unsqueeze(0) for key, value in observations.items()},
        "action": cast("torch.Tensor", sample["action"]).unsqueeze(0),
    }


def _parameter_snapshot(policy: BaseImagePolicy) -> list[torch.Tensor]:
    return [
        parameter.detach().clone() for parameter in policy.parameters() if parameter.requires_grad
    ]


def _parameter_delta(policy: BaseImagePolicy, before: list[torch.Tensor]) -> float:
    current = [parameter.detach() for parameter in policy.parameters() if parameter.requires_grad]
    return float(
        sum((new - old).abs().sum().item() for new, old in zip(current, before, strict=True))
    )


def _gradient_squared_norm(policy: BaseImagePolicy) -> float:
    return float(
        sum(
            parameter.grad.detach().square().sum().item()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
    )


def _execute_update(
    policy: BaseImagePolicy,
    model: str,
    batch: dict[str, object],
    optimizer: object | None,
) -> _UpdateMetrics:
    before = _parameter_snapshot(policy)
    gradient_norm = 0.0
    if model == "lstm_gmm":
        info = cast("_RobomimicTrainPolicy", policy).train_on_batch(batch, epoch=0)
        losses_raw = info.get("losses")
        if not isinstance(losses_raw, dict):
            raise RuntimeError("upstream LSTM-GMM did not return losses")
        loss_tensor: object = cast("dict[str, object]", losses_raw).get("action_loss")
    else:
        compute_loss = getattr(policy, "compute_loss", None)
        zero_grad = getattr(optimizer, "zero_grad", None)
        step = getattr(optimizer, "step", None)
        if not callable(compute_loss) or not callable(zero_grad) or not callable(step):
            raise TypeError("locked workspace policy or optimizer lifecycle is incomplete")
        zero_grad(set_to_none=True)
        loss_tensor = cast("Callable[[dict[str, object]], object]", compute_loss)(batch)
        if not isinstance(loss_tensor, torch.Tensor) or loss_tensor.ndim != 0:
            raise RuntimeError("upstream policy loss must be a scalar tensor")
        cast("Callable[[], object]", loss_tensor.backward)()
        gradient_norm = _gradient_squared_norm(policy)
        cast("Callable[[], object]", step)()
    if not isinstance(loss_tensor, torch.Tensor):
        raise TypeError("upstream policy did not return a tensor loss")
    if model == "lstm_gmm":
        gradient_norm = _gradient_squared_norm(policy)
    loss = float(loss_tensor.detach().cpu().item())
    delta = _parameter_delta(policy, before)
    if (
        not math.isfinite(loss)
        or not math.isfinite(gradient_norm)
        or gradient_norm <= 0
        or not math.isfinite(delta)
        or delta <= 0
    ):
        raise RuntimeError(
            "one-batch optimizer update did not produce finite gradients and parameters"
        )
    return _UpdateMetrics(loss, gradient_norm, delta)


def _prepare_policy_batch(
    policy: BaseImagePolicy, model: str, store: Path, mode: SmokeMode
) -> dict[str, object]:
    split = "synthetic_probe" if mode == "fixture" else "train"
    batch = _batch(store, model, split)
    dataset = PaperBaselineDataset(store, horizon=PROFILES[model].horizon, split=split)
    set_normalizer = getattr(policy, "set_normalizer", None)
    if not callable(set_normalizer):
        raise TypeError("locked policy has no normalizer seam")
    cast("Callable[[object], None]", set_normalizer)(dataset.get_normalizer())
    cast("Callable[[], object]", policy.train)()
    return batch


def _smoke_receipt(
    model: str,
    identity: SmokeStoreIdentity,
    policy: BaseImagePolicy,
    metrics: _UpdateMetrics,
) -> dict[str, object]:
    policy_origin, workspace_origin = validate_model_identity(model, policy)
    profile = PROFILES[model]
    return {
        "model": model,
        "mode": identity.mode,
        "policy_class": type(policy).__name__,
        "policy_module": type(policy).__module__,
        "policy_origin": str(policy_origin),
        "workspace_class": profile.workspace_target.rsplit(".", 1)[1],
        "workspace_module": profile.workspace_target.rsplit(".", 1)[0],
        "workspace_origin": str(workspace_origin),
        "stanford_commit": _STANFORD_COMMIT,
        "robomimic_commit": _ROBOMIMIC_COMMIT,
        "observation_steps": profile.observation_steps,
        "horizon": profile.horizon,
        "executed_actions": profile.executed_actions,
        "action_dim": SHAPE_META["action"]["shape"][0],
        "canonical_digest": identity.canonical_digest,
        "split_digest": identity.split_digest,
        "training_eligible": identity.training_eligible,
        "comparison_eligible": identity.comparison_eligible,
        "result_status": (
            "ineligible_fixture"
            if identity.mode == "fixture"
            else "production_smoke_complete_nonfinal"
        ),
        "optimizer_steps": 1,
        "loss": metrics.loss,
        "gradient_squared_norm": metrics.gradient_squared_norm,
        "parameter_delta": metrics.parameter_delta,
        "recurrent_identity": (
            "BC_RNN_GMM/RNNGMMActorNetwork/LSTM/GMM" if model == "lstm_gmm" else None
        ),
    }


def run_workspace_one_batch(
    workspace: object,
    model: str,
    store: str | Path,
    *,
    mode: SmokeMode,
) -> dict[str, object]:
    """Run one real update through a constructed pinned Stanford workspace."""
    path = Path(store)
    identity = validate_smoke_store(path, mode=mode)
    policy = getattr(workspace, "model", None)
    if not isinstance(policy, BaseImagePolicy):
        raise TypeError("locked workspace did not construct an upstream policy")
    validate_model_identity(model, policy)
    optimizer = None if model == "lstm_gmm" else getattr(workspace, "optimizer", None)
    batch = _prepare_policy_batch(policy, model, path, mode)
    metrics = _execute_update(policy, model, batch, optimizer)
    values = getattr(workspace, "__dict__", None)
    if isinstance(values, dict):
        cast("dict[str, object]", values)["global_step"] = 1
    return _smoke_receipt(model, identity, policy, metrics)


def run_one_batch_smoke(
    model: str, store: str | Path, *, mode: SmokeMode = "fixture"
) -> dict[str, object]:
    """Instantiate one real policy and execute exactly one upstream-style optimizer update."""
    if model not in PROFILES:
        raise ValueError(f"unknown paper baseline: {model}")
    path = Path(store)
    identity = validate_smoke_store(path, mode=mode)
    cast("Callable[[int], object]", torch.manual_seed)(0)
    policy_object = instantiate(OmegaConf.create(bounded_policy_config(model)))
    if not isinstance(policy_object, BaseImagePolicy):
        raise TypeError("Hydra did not instantiate an upstream BaseImagePolicy")
    policy = policy_object
    validate_model_identity(model, policy)
    batch = _prepare_policy_batch(policy, model, path, mode)
    optimizer = None if model == "lstm_gmm" else torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = _execute_update(policy, model, batch, optimizer)
    return _smoke_receipt(model, identity, policy, metrics)
