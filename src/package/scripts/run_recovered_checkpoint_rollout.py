#!/usr/bin/env python3
"""Run one local MuJoCo rollout from a verified recovered full checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=100000, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_legacy_local_runtime(
    config: dict[str, object], *, optimizer_updates: int
) -> None:
    expected_shape_meta = {
        "obs": {
            "cam_top": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [5], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
    if config.get("shape_meta") != expected_shape_meta or optimizer_updates != 400_000:
        raise RuntimeError("recovered checkpoint is not the supported 96px local profile")
    for variable in ("PUSHT_SINGLE_CAM", "PUSHT_LOCAL_BUDGET"):
        existing = os.environ.get(variable)
        if existing not in (None, "1"):
            raise RuntimeError(f"{variable} must be unset or '1' for this checkpoint")
        os.environ[variable] = "1"


def load_policy(root: Path) -> tuple[object, object, str]:
    receipt_path = root / "training_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"missing training receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "pusht-so100-full-training-v1"
        or receipt.get("completed") is not True
        or receipt.get("training_mode") != "full_production"
    ):
        raise RuntimeError("checkpoint root does not contain a completed full-training receipt")
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    optimizer_updates = receipt.get("executed_optimizer_updates")
    if type(optimizer_updates) is not int:
        raise RuntimeError("checkpoint receipt optimizer update count is invalid")
    configure_legacy_local_runtime(
        config,
        optimizer_updates=optimizer_updates,
    )
    from omegaconf import OmegaConf

    from so101_pusht_benchmark.training.identity import BundleIdentity

    identity = BundleIdentity.from_dict(receipt.get("identity"))
    checkpoint = root / "checkpoints/latest.ckpt"
    module_name, class_name = identity.workspace_target.rsplit(".", maxsplit=1)
    workspace_class = getattr(import_module(module_name), class_name)
    workspace = workspace_class(OmegaConf.create(config), output_dir=str(root))
    workspace.load_checkpoint(path=checkpoint)
    ema_model = getattr(workspace, "ema_model", None)
    policy = ema_model if ema_model is not None else workspace.model
    policy_module, policy_class = identity.policy_target.rsplit(".", maxsplit=1)
    if not isinstance(policy, getattr(import_module(policy_module), policy_class)):
        raise RuntimeError("recovered checkpoint loaded an unexpected policy class")
    policy.to("cuda:0")
    policy.eval()
    return policy, identity, sha256(checkpoint)


def main() -> int:
    args = parse_args()
    try:
        policy, identity, checkpoint_sha256 = load_policy(args.checkpoint_root.resolve())
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    from generate_feedback_artifacts import CaptureOptions, capture_rollout

    capture = capture_rollout(
        policy,
        identity,
        args.seed,
        "RECOVERED DP-CNN",
        CaptureOptions(render=False),
    )
    result = {
        "model": identity.model,
        "seed": args.seed,
        "success": capture.success,
        "steps": len(capture.targets) - 1,
        "checkpoint_sha256": checkpoint_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
