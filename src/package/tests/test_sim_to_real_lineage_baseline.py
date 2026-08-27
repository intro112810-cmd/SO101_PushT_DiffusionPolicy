from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from so101_pusht_benchmark.training.artifacts import ArtifactIndex

_ARTIFACT_ROOT = Path("/home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark")
_ARTIFACT_ID = "local-dp_cnn-recovered-v3-seed0"


def test_current_bundle_validator_and_runtime_identity_baseline() -> None:
    index = ArtifactIndex(_ARTIFACT_ROOT / "artifact-index.json", _ARTIFACT_ROOT)
    record = index.record(_ARTIFACT_ID)
    manifest_path = index.verify(_ARTIFACT_ID, "manifest")
    assert index.verify(_ARTIFACT_ID, "bundle").name == "policy.safetensors"
    assert index.verify(_ARTIFACT_ID, "normalizer").name == "normalizer.json"
    assert index.verify(_ARTIFACT_ID, "config").name == "resolved_config.json"
    assert index.verify(_ARTIFACT_ID, "checkpoint").name == "latest.ckpt"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = cast("dict[str, object]", record["identity"])
    assert manifest["identity"] == identity
    assert identity["model"] == "dp_cnn"
    assert (
        identity["optimizer_updates"],
        identity["observation_steps"],
        identity["horizon"],
        identity["executed_actions"],
    ) == (400_000, 2, 16, 8)
    assert identity["runtime_lock_digest"] == (
        "10776208a02c73299caf78249cecd8d1d6870e026ab456ec1c11087adc521b9a"
    )
