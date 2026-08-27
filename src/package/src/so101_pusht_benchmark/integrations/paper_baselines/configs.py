"""Locked four-model profiles that target unchanged upstream implementations.

Original sources are Stanford commit 5ba07ac6661db573af695b419a7947ecb704690f
and robomimic archive commit 62ed2de905caeb9133136e4d14d810a8b6baa96c.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
from typing import Protocol, TypedDict, cast

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# robomimic emits an import-time macro diagnostic with print(). Keep dependency
# diagnostics visible while preserving stdout for machine-readable CLI results.
with redirect_stdout(sys.stderr):
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
        DiffusionTransformerHybridImagePolicy,
    )
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )
    from diffusion_policy.policy.ibc_dfo_hybrid_image_policy import IbcDfoHybridImagePolicy
    from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy

    from .robomimic import assert_lstm_gmm_runtime


@dataclass(frozen=True, slots=True)
class ModelProfile:
    policy_class: type[BaseImagePolicy]
    workspace_target: str
    observation_steps: int
    horizon: int
    executed_actions: int
    batch_size: int = 64
    optimizer_updates: int = 100_000
    train_steps_per_epoch: int = 5_000
    epochs: int = 20


PROFILES = {
    "dp_cnn": ModelProfile(
        DiffusionUnetHybridImagePolicy,
        "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace",
        2,
        16,
        8,
    ),
    "dp_transformer": ModelProfile(
        DiffusionTransformerHybridImagePolicy,
        "diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace",
        2,
        16,
        8,
    ),
    "ibc": ModelProfile(
        IbcDfoHybridImagePolicy,
        "diffusion_policy.workspace.train_ibc_dfo_hybrid_workspace.TrainIbcDfoHybridWorkspace",
        2,
        2,
        1,
    ),
    "lstm_gmm": ModelProfile(
        RobomimicImagePolicy,
        "diffusion_policy.workspace.train_robomimic_image_workspace.TrainRobomimicImageWorkspace",
        10,
        10,
        1,
    ),
}


class _ShapeField(TypedDict):
    shape: list[int]
    type: str


class _ObservationShape(TypedDict):
    cam_top: _ShapeField
    cam_side: _ShapeField
    agent_pos: _ShapeField


class _SingleCamObservationShape(TypedDict):
    cam_top: _ShapeField
    agent_pos: _ShapeField


class _ActionShape(TypedDict):
    shape: list[int]


class ShapeMeta(TypedDict):
    obs: _ObservationShape | _SingleCamObservationShape
    action: _ActionShape


_SINGLE_CAM = os.environ.get("PUSHT_SINGLE_CAM") == "1"

if _SINGLE_CAM:
    SHAPE_META: ShapeMeta = {
        "obs": {
            "cam_top": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [5], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
else:
    SHAPE_META: ShapeMeta = {
        "obs": {
            "cam_top": {"shape": [3, 224, 224], "type": "rgb"},
            "cam_side": {"shape": [3, 224, 224], "type": "rgb"},
            "agent_pos": {"shape": [5], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }


class PolicyNamespaceError(TypeError):
    """A model config or encoder does not expose the native policy namespace."""


class _NestedNets(Protocol):
    nets: dict[str, object]


class _PolicyNets(Protocol):
    nets: dict[str, _NestedNets]


class _RobomimicPolicy(Protocol):
    nets: dict[str, _PolicyNets]


def _exact_shape(value: object, expected: list[int]) -> bool:
    if not isinstance(value, list):
        return False
    items = cast("list[object]", value)
    return all(type(item) is int for item in items) and items == expected


def validate_shape_meta(shape_meta: object) -> None:
    """Reject any policy shape metadata other than the exact native namespace."""
    if not isinstance(shape_meta, dict):
        raise PolicyNamespaceError("shape_meta keys/order mismatch")
    root = cast("dict[str, object]", shape_meta)
    if tuple(root) != ("obs", "action"):
        raise PolicyNamespaceError("shape_meta keys/order mismatch")
    observation = root["obs"]
    action = root["action"]
    if not isinstance(observation, dict):
        raise PolicyNamespaceError("shape_meta observation keys/order mismatch")
    observation_fields = cast("dict[str, object]", observation)
    if _SINGLE_CAM:
        if tuple(observation_fields) != ("cam_top", "agent_pos"):
            raise PolicyNamespaceError("shape_meta observation keys/order mismatch")
        expected = {
            "cam_top": ([3, 96, 96], "rgb"),
            "agent_pos": ([5], "low_dim"),
        }
    else:
        if tuple(observation_fields) != (
            "cam_top",
            "cam_side",
            "agent_pos",
        ):
            raise PolicyNamespaceError("shape_meta observation keys/order mismatch")
        expected = {
            "cam_top": ([3, 224, 224], "rgb"),
            "cam_side": ([3, 224, 224], "rgb"),
            "agent_pos": ([5], "low_dim"),
        }
    for key, (shape, modality) in expected.items():
        field = observation_fields[key]
        if not isinstance(field, dict):
            raise PolicyNamespaceError(f"shape_meta {key} fields mismatch")
        values = cast("dict[str, object]", field)
        if tuple(values) != ("shape", "type"):
            raise PolicyNamespaceError(f"shape_meta {key} fields mismatch")
        if not _exact_shape(values["shape"], shape) or values["type"] != modality:
            raise PolicyNamespaceError(f"shape_meta {key} shape/type mismatch")
    if not isinstance(action, dict):
        raise PolicyNamespaceError("shape_meta action must be float32[2]")
    action_values = cast("dict[str, object]", action)
    if tuple(action_values) != ("shape",) or not _exact_shape(action_values["shape"], [2]):
        raise PolicyNamespaceError("shape_meta action must be float32[2]")


def observation_encoder(policy: object) -> object:
    """Resolve and validate the real robomimic encoder built by a Stanford policy."""
    if not isinstance(policy, tuple(profile.policy_class for profile in PROFILES.values())):
        raise PolicyNamespaceError("policy class is not a locked Stanford implementation")
    if isinstance(
        policy,
        (
            DiffusionUnetHybridImagePolicy,
            DiffusionTransformerHybridImagePolicy,
            IbcDfoHybridImagePolicy,
        ),
    ):
        encoder = policy.obs_encoder
    else:
        robomimic_policy = cast("_RobomimicPolicy", policy)
        encoder = robomimic_policy.nets["policy"].nets["encoder"].nets["obs"]
    module = type(encoder).__module__
    obs_nets = getattr(encoder, "obs_nets", None)
    if module != "robomimic.models.obs_nets" or not isinstance(obs_nets, torch.nn.ModuleDict):
        raise PolicyNamespaceError(
            "observation encoder is not upstream robomimic ObservationEncoder"
        )
    expected_obs_nets = ("cam_top", "agent_pos") if _SINGLE_CAM else ("cam_top", "cam_side", "agent_pos")
    if tuple(obs_nets) != expected_obs_nets:
        raise PolicyNamespaceError("observation encoder keys/order mismatch")
    project_root = Path(__file__).resolve().parents[6]
    policy_source = Path(inspect.getfile(type(policy))).resolve()
    encoder_source = Path(inspect.getfile(type(encoder))).resolve()
    expected_policy = (
        project_root
        / "04_experiments/so101_pusht_benchmark/cache/upstream/stanford/diffusion_policy/policy"
        / policy_source.name
    )
    expected_encoder = (
        project_root
        / "04_experiments/so101_pusht_benchmark/cache/upstream/robomimic/robomimic/models/obs_nets.py"
    )
    if (
        not expected_policy.is_file()
        or policy_source.read_bytes() != expected_policy.read_bytes()
        or not expected_encoder.is_file()
        or encoder_source.read_bytes() != expected_encoder.read_bytes()
    ):
        raise PolicyNamespaceError("policy or observation encoder does not match pinned upstream")
    return encoder


def _scheduler() -> dict[str, object]:
    return {
        "_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "variance_type": "fixed_small",
        "clip_sample": True,
        "prediction_type": "epsilon",
    }


def policy_config(name: str) -> dict[str, object]:
    validate_shape_meta(SHAPE_META)
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {name}")
    common: dict[str, object] = {
        "_target_": f"{profile.policy_class.__module__}.{profile.policy_class.__name__}",
        "shape_meta": SHAPE_META,
    }
    if name == "lstm_gmm":
        return {
            **common,
            "algo_name": "bc_rnn",
            "obs_type": "image",
            "task_name": "square",
            "dataset_type": "ph",
            "crop_shape": [76, 76],
        }
    common.update(
        {
            "horizon": profile.horizon,
            "n_action_steps": profile.executed_actions,
            "n_obs_steps": profile.observation_steps,
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
        }
    )
    if name == "ibc":
        common.update(
            {
                "dropout": 0.1,
                "train_n_neg": 1024,
                "pred_n_iter": 5,
                "pred_n_samples": 1024,
                "crop_shape": [84, 84],
            }
        )
    else:
        common.update({"noise_scheduler": _scheduler(), "num_inference_steps": 100})
        common["crop_shape"] = [76, 76]
        if name == "dp_cnn":
            common.update(
                {
                    "obs_as_global_cond": True,
                    "diffusion_step_embed_dim": 128,
                    "down_dims": [512, 1024, 2048],
                    "cond_predict_scale": True,
                }
            )
    return common


def workspace_config(name: str, paper_view: str | Path, seed: int) -> dict[str, object]:
    """Build a Hydra config accepted by the corresponding unchanged workspace."""
    validate_shape_meta(SHAPE_META)
    if seed not in (0, 1, 2):
        raise ValueError("training seed must be one of 0, 1, 2")
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown paper baseline: {name}")
    dataloader_batch_size = 32
    if name == "lstm_gmm":
        raw_batch_size = os.environ.get("PUSHT_LSTM_BATCH_SIZE", "32")
        try:
            dataloader_batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError("PUSHT_LSTM_BATCH_SIZE must be an integer") from exc
        if dataloader_batch_size < 1:
            raise ValueError("PUSHT_LSTM_BATCH_SIZE must be positive")
    recurrent = name == "lstm_gmm"
    pad_before = 0 if recurrent else profile.observation_steps - 1
    pad_after = 0 if recurrent else profile.executed_actions - 1
    config: dict[str, object] = {
        "_target_": profile.workspace_target,
        "_recursive_": False,
        "name": name,
        "shape_meta": SHAPE_META,
        "horizon": profile.horizon,
        "n_obs_steps": profile.observation_steps,
        "n_action_steps": profile.executed_actions,
        "policy": policy_config(name),
        "task": {
            "dataset": {
                "_target_": "so101_pusht_benchmark.integrations.paper_baselines.dataset.PaperBaselineDataset",
                "zarr_path": str(paper_view),
                "horizon": profile.horizon,
                "pad_before": pad_before,
                "pad_after": pad_after,
            },
            "env_runner": {
                "_target_": "so101_pusht_benchmark.integrations.paper_baselines.runner.PaperBaselineRunner",
                "evaluation_seeds": list(range(100000, 100100)),
                "n_obs_steps": profile.observation_steps,
                "n_action_steps": profile.executed_actions,
                "options": {"native_env_factory": None},
            },
        },
        "dataloader": {
            "batch_size": dataloader_batch_size,
            "shuffle": True,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": dataloader_batch_size,
            "shuffle": False,
            "num_workers": 0,
        },
        "training": {
            "seed": seed,
            "device": os.environ.get("PUSHT_DEVICE", "cuda:0"),
            "debug": False,
            "resume": True,
            "num_epochs": 1,
            "max_train_steps": (
                400_000 if os.environ.get("PUSHT_LOCAL_BUDGET") == "1" else 100_000
            ),
            "max_val_steps": None,
            "rollout_every": 1,
            "checkpoint_every": 19,
            "val_every": 1,
            "sample_every": 1,
            "sample_max_batch": 64,
            "tqdm_interval_sec": 1.0,
            "use_ema": name in {"dp_cnn", "dp_transformer"},
            "gradient_accumulate_every": 1,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 500,
        },
        "logging": {"project": "so101_pusht", "mode": "offline", "name": name},
        "checkpoint": {
            "topk": {
                "monitor_key": "val_loss",
                "mode": "min",
                "k": 1,
                "format_str": "epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt",
            },
            "save_last_ckpt": True,
            "save_last_snapshot": False,
        },
    }
    if name == "dp_transformer":
        training = config["training"]
        if isinstance(training, dict):
            training["lr_warmup_steps"] = 1000
        config["optimizer"] = {
            "transformer_weight_decay": 1e-3,
            "obs_encoder_weight_decay": 1e-6,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.95],
        }
    elif name != "lstm_gmm":
        config["optimizer"] = {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-4,
            "betas": [0.95, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-6,
        }
    if name in {"dp_cnn", "dp_transformer"}:
        config["ema"] = {
            "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,
            "min_value": 0.0,
            "max_value": 0.9999,
        }
    return config


def instantiate_policy(name: str) -> BaseImagePolicy:
    config = policy_config(name)
    validate_shape_meta(config["shape_meta"])
    policy = instantiate(OmegaConf.create(config))
    if not isinstance(policy, PROFILES[name].policy_class):
        raise TypeError("Hydra did not instantiate the locked upstream policy class")
    result = policy
    observation_encoder(result)
    if name == "lstm_gmm":
        assert_lstm_gmm_runtime(result)
    return result
