"""Export any pinned model checkpoint to a digest-bound tensor-only inference bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Callable
from typing import Protocol, cast

from .artifacts import (
    ArtifactError,
    ArtifactIndex,
    ArtifactScope,
    BundleFiles,
    sha256_file,
)
from .identity import BundleIdentity


def assert_paper_runtime() -> None:
    """Import and verify the pinned runtime only after artifact authentication."""
    from .runtime import assert_paper_runtime as verify

    verify()


def resolve_workspace_class(model: str) -> type[object]:
    """Resolve imported upstream code only after artifact authentication."""
    from .model_smoke import resolve_workspace_class as resolve

    return resolve(model)


class _ReloadWorkspace(Protocol):
    model: object

    def load_checkpoint(self, *, path: Path) -> None: ...


def export_inference_bundle(
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    *,
    artifact_id: str,
    index: ArtifactIndex,
) -> Path:
    """Verify native checkpoint identity before reload and atomically publish tensors."""
    anchored_checkpoint, anchored_config = index.require_trusted_production_checkpoint(artifact_id)
    record = index.record(artifact_id)
    if anchored_checkpoint != checkpoint.resolve():
        raise ArtifactError("checkpoint argument does not match anchored path")
    if anchored_config != config_path.resolve():
        raise ArtifactError("config argument does not match anchored path")
    identity = BundleIdentity.from_dict(record.get("identity"))
    assert_paper_runtime()
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    from omegaconf import OmegaConf

    from .bundle import save_bundle
    from .metadata import read_trusted_config
    from .model_smoke import validate_model_identity

    config = read_trusted_config(anchored_config, identity.model)
    workspace_class = resolve_workspace_class(identity.model)

    output_dir = output_dir.absolute()
    try:
        output_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ArtifactError(f"output already exists: {output_dir}")
    token = hashlib.sha256(f"bundle:{artifact_id}".encode()).hexdigest()[:12]
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{token}")
    staging = index.create_output_directory(staging)
    published = False
    try:
        factory = cast("Callable[..., _ReloadWorkspace]", workspace_class)
        workspace = factory(OmegaConf.create(config), output_dir=str(staging))
        workspace.load_checkpoint(path=anchored_checkpoint)
        ema_model = getattr(workspace, "ema_model", None)
        candidate = ema_model if ema_model is not None else workspace.model
        if not isinstance(candidate, BaseImagePolicy):
            raise ArtifactError("checkpoint did not reload a pinned upstream policy")
        policy = candidate
        validate_model_identity(identity.model, policy)
        state = dict(policy.state_dict())
        bundle = staging / "policy.safetensors"
        save_bundle(bundle, state)
        resolved = staging / "resolved_config.json"
        resolved.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        normalizer_keys = sorted(key for key in state if "normalizer" in key)
        metadata = {
            "schema": 1,
            "deployment_scope": "simulation_only",
            "training_eligible": record.get("training_eligible") is True,
            "source_checkpoint_sha256": sha256_file(anchored_checkpoint),
            "resolved_config_sha256": sha256_file(anchored_config),
            "identity": identity.to_dict(),
            "state": {
                key: {"shape": list(state[key].shape), "dtype": str(state[key].dtype)}
                for key in normalizer_keys
            },
        }
        normalizer = staging / "normalizer.json"
        normalizer.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = staging / "bundle_manifest.json"
        manifest.write_text(
            json.dumps(
                identity.bundle_manifest(anchored_checkpoint, anchored_config),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
        published = True
        final_bundle = output_dir / bundle.name
        index.anchor_bundle(
            artifact_id,
            BundleFiles(
                final_bundle,
                output_dir / resolved.name,
                output_dir / normalizer.name,
                output_dir / manifest.name,
            ),
            ArtifactScope(
                training_mode="full_production",
                identity=identity.to_dict(),
            ),
        )
    except BaseException:
        shutil.rmtree(output_dir if published else staging, ignore_errors=True)
        raise
    else:
        return final_bundle
