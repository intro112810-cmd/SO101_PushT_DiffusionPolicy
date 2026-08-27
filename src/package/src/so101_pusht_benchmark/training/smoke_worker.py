"""Paper-runtime worker for Todo 10 training, strict reload, and policy inference."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
from copy import deepcopy
import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from ..data.paper_view_reader import load_paper_view
from ..integrations.paper_baselines.configs import PROFILES, SHAPE_META
from ..integrations.paper_baselines.runner import PaperBaselineRunner
from .artifacts import ArtifactError, ArtifactIndex, sha256_file
from .identity import BundleIdentity, trusted_identity
from .launcher import TrainingLaunch, launch_training
from .model_smoke import resolve_workspace_class, validate_model_identity, validate_profile_config
from .smoke_contract import MODEL_SMOKE_SCHEMA


class _ReloadWorkspace(Protocol):
    model: object

    def load_checkpoint(self, *, path: Path) -> None: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="todo10-model-smoke-worker")
    parser.add_argument("phase", choices=("train", "infer"))
    parser.add_argument("--model", choices=tuple(PROFILES), required=True)
    parser.add_argument("--expected-model", choices=tuple(PROFILES), required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def _index(root: Path) -> tuple[ArtifactIndex, str]:
    path = root / "artifact-index.json"
    artifact_id = "fixture-model-smoke"
    if not path.exists():
        path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    return ArtifactIndex(path, root), artifact_id


def _ordered_config(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError("resolved smoke config must be an object")
    config = cast("dict[str, object]", value)
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ArtifactError("resolved smoke policy config is missing")
    config["shape_meta"] = deepcopy(SHAPE_META)
    cast("dict[str, object]", policy)["shape_meta"] = deepcopy(SHAPE_META)
    return config


def _strict_reload(
    root: Path, expected_model: str
) -> tuple[BaseImagePolicy, BundleIdentity, Path, Path, dict[str, object]]:
    index, artifact_id = _index(root)
    record = index.record(artifact_id)
    identity = BundleIdentity.from_dict(record.get("identity"))
    expected = trusted_identity(expected_model, identity.dataset_digest, identity.split_digest)
    if identity != expected:
        raise ArtifactError("trusted model identity mismatch before environment creation")
    checkpoint = index.verify(artifact_id, "checkpoint")
    config_path = index.verify(artifact_id, "config")
    config = _ordered_config(config_path)
    validate_profile_config(expected_model, config)
    workspace_class = resolve_workspace_class(expected_model)
    if config.get("_target_") != f"{workspace_class.__module__}.{workspace_class.__name__}":
        raise ArtifactError("resolved config workspace identity mismatch")
    factory = cast("Callable[..., _ReloadWorkspace]", workspace_class)
    workspace = factory(OmegaConf.create(config), output_dir=str(root / "reload-workspace"))
    workspace.load_checkpoint(path=checkpoint)
    if not isinstance(workspace.model, BaseImagePolicy):
        raise ArtifactError("checkpoint reload did not restore an upstream policy")
    policy = workspace.model
    validate_model_identity(expected_model, policy)
    return policy, identity, checkpoint, config_path, config


def _compact_identity(identity: BundleIdentity) -> dict[str, object]:
    return {
        "model": identity.model,
        "policy_target": identity.policy_target,
        "workspace_target": identity.workspace_target,
        "observation_steps": identity.observation_steps,
        "horizon": identity.horizon,
        "executed_actions": identity.executed_actions,
        "optimizer_updates": identity.optimizer_updates,
        "dataset_digest": identity.dataset_digest,
        "split_digest": identity.split_digest,
        "runtime_lock_digest": identity.runtime_lock_digest,
        "environment_manifest_digest": identity.environment_manifest_digest,
        "stanford_commit": identity.stanford_commit,
        "robomimic_commit": identity.robomimic_commit,
    }


def _train(args: argparse.Namespace) -> None:
    index, artifact_id = _index(args.root)
    checkpoint = launch_training(
        args.store,
        args.root / "training",
        index,
        TrainingLaunch(
            seed=0,
            artifact_id=artifact_id,
            model=args.model,
            smoke_mode="fixture",
        ),
    )
    policy, identity, anchored_checkpoint, config_path, _ = _strict_reload(
        args.root, args.expected_model
    )
    if checkpoint.resolve() != anchored_checkpoint:
        raise ArtifactError("checkpoint reload path identity mismatch")
    smoke_value: object = json.loads(
        (args.root / "training/smoke_receipt.json").read_text(encoding="utf-8")
    )
    if not isinstance(smoke_value, dict):
        raise ArtifactError("optimizer smoke receipt is malformed")
    smoke = cast("dict[str, object]", smoke_value)
    view = load_paper_view(args.store)
    canonical_digest = view.manifest.get("canonical_digest")
    root_digest = view.manifest.get("root_digest")
    if canonical_digest != identity.dataset_digest or not isinstance(root_digest, str):
        raise ArtifactError("reloaded identity does not match the validated store")
    receipt: dict[str, object] = {
        "schema": MODEL_SMOKE_SCHEMA,
        "phase": "strict_reload",
        "model": args.expected_model,
        "fixture": True,
        "production_eligible": False,
        "comparison_eligible": False,
        "identity": _compact_identity(identity),
        "store_identity": {
            "canonical_digest": canonical_digest,
            "root_digest": root_digest,
            "split_digest": identity.split_digest,
            "manifest_sha256": sha256_file(args.store / "manifest.json"),
            "splits_sha256": sha256_file(args.store / "splits.json"),
        },
        "checkpoint": anchored_checkpoint.relative_to(args.root).as_posix(),
        "checkpoint_sha256": sha256_file(anchored_checkpoint),
        "config": config_path.relative_to(args.root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "reload_verified": True,
        "policy_class": type(policy).__name__,
        "optimizer_updates": smoke.get("optimizer_steps"),
        "loss": smoke.get("loss"),
    }
    (args.root / "reload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_inference(args.root, policy, identity)


def _observation(root: Path) -> dict[str, NDArray[np.generic]]:
    values: dict[str, NDArray[np.generic]] = {}
    for key in ("cam_top", "cam_side", "agent_pos"):
        value = np.load(root / f"rollout-{key}.npy", allow_pickle=False)
        values[key] = cast("NDArray[np.generic]", value)
    return values


def _write_inference(root: Path, policy: BaseImagePolicy, identity: BundleIdentity) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    policy.reset()
    observation = _observation(root)
    history = deque(
        (observation for _ in range(identity.observation_steps)),
        maxlen=identity.observation_steps,
    )
    with torch.no_grad():
        prediction = policy.predict_action(PaperBaselineRunner.policy_observation(history, policy))
    action_tensor = prediction.get("action")
    if not isinstance(action_tensor, torch.Tensor):
        raise ArtifactError("reloaded policy did not return an action tensor")
    to_numpy = cast("Callable[[], NDArray[np.generic]]", action_tensor.detach().cpu().numpy)
    action = to_numpy()
    if action.ndim != 3 or action.shape[0] != 1 or action.shape[2] != 2 or action.shape[1] < 1:
        raise ArtifactError("reloaded policy action shape is not [1,T,2]")
    selected = np.ascontiguousarray(action[0, 0], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ArtifactError("reloaded policy action is non-finite")
    if bool(np.any(selected < -1.0)) or bool(np.any(selected > 1.0)):
        raise ArtifactError("reloaded policy action exceeds native bounds; clipping is forbidden")
    selected.tofile(root / "action.bin")
    (root / "inference.json").write_text(
        json.dumps(
            {
                "schema": MODEL_SMOKE_SCHEMA,
                "model": identity.model,
                "checkpoint_reloaded": True,
                "action_dtype": "float32",
                "action_shape": [2],
                "action": selected.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _infer(args: argparse.Namespace) -> None:
    policy, identity, _, _, _ = _strict_reload(args.root, args.expected_model)
    _write_inference(args.root, policy, identity)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "train":
        _train(args)
    else:
        _infer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
