from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
import torch

from so101_pusht_benchmark.integrations.paper_baselines.configs import workspace_config
from so101_pusht_benchmark.training.artifacts import (
    ArtifactError,
    ArtifactIndex,
    ArtifactScope,
    BundleFiles,
    sha256_file,
)
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle, save_bundle
from so101_pusht_benchmark.training.identity import trusted_identity
from so101_pusht_benchmark.training.launcher import (
    configure_cuda_runtime,
    full_production_config,
    full_production_sample_count,
    resolved_dp_cnn_config,
    update_budget,
    remaining_full_production_sample_count,
    restore_workspace_output_dir,
)


def test_configure_cuda_runtime_disables_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUSHT_DISABLE_CUDNN", "1")
    previous = torch.backends.cudnn.enabled
    try:
        configure_cuda_runtime()
        assert torch.backends.cudnn.enabled is False
    finally:
        torch.backends.cudnn.enabled = previous


def test_workspace_config_honors_scoped_device_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PUSHT_DEVICE", "cpu")

    config = workspace_config("lstm_gmm", tmp_path / "paper-view", seed=0)

    assert config["training"]["device"] == "cpu"


def test_dp_cnn_resolution_maps_exact_update_budget(tmp_path: Path) -> None:
    config = resolved_dp_cnn_config(tmp_path / "paper-view", seed=1)
    assert config["_recursive_"] is False
    assert config["_target_"] == (
        "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace."
        "TrainDiffusionUnetHybridWorkspace"
    )
    training = cast("dict[str, object]", config["training"])
    assert training["max_train_steps"] == 5_000
    assert training["num_epochs"] == 20
    assert update_budget(config) == 100_000
    assert cast("dict[str, object]", config["dataloader"])["batch_size"] == 64
    assert all(training[f"{name}_every"] == 1 for name in ("val", "checkpoint", "rollout"))
    policy = cast("dict[str, object]", config["policy"])
    assert policy["down_dims"] == [512, 1024, 2048]


def test_synthetic_probe_is_bounded_without_mutating_production_budget(tmp_path: Path) -> None:
    from so101_pusht_benchmark.training.launcher import synthetic_probe_config

    production = resolved_dp_cnn_config(tmp_path / "paper-view", seed=1)
    probe = synthetic_probe_config(production)
    assert update_budget(production) == 100_000
    assert update_budget(probe) == 30
    assert cast("dict[str, object]", production["policy"])["down_dims"] == [512, 1024, 2048]
    assert cast("dict[str, object]", probe["policy"])["down_dims"] == [32, 64, 128]
    dataset = cast("dict[str, object]", cast("dict[str, object]", probe["task"])["dataset"])
    assert dataset["split"] == "synthetic_probe"


def test_full_production_keeps_exact_budget_and_defers_rollout() -> None:
    config = resolved_dp_cnn_config(Path("/frozen"), 0)
    production = full_production_config(config)
    training = cast("dict[str, object]", production["training"])
    assert update_budget(production) == 100_000
    assert training["resume"] is False
    assert training["num_epochs"] == 1
    assert training["max_train_steps"] == 100_000
    assert full_production_sample_count(production) == 100_000 * 32
    assert training["rollout_every"] == 2
    assert cast("dict[str, object]", config["training"])["rollout_every"] == 1


def test_resume_uses_only_remaining_full_production_samples() -> None:
    config = full_production_config(resolved_dp_cnn_config(Path("/frozen"), 0))

    assert remaining_full_production_sample_count(config, 50_000) == 50_000 * 32
    with pytest.raises(ArtifactError, match="already reached"):
        remaining_full_production_sample_count(config, 100_000)


def test_resume_restores_workspace_output_directory() -> None:
    class Workspace:
        _output_dir = "/source-machine/staging"

    workspace = Workspace()
    restore_workspace_output_dir(workspace, Path("/server/staging"))

    assert workspace._output_dir == "/server/staging"


def test_tensor_bundle_strict_shapes_and_hash_anchor(
    tmp_path: Path, secure_test_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_pusht_benchmark.training import artifacts

    class TestAccount:
        pw_dir = str(secure_test_home)

    def account_lookup(_uid: int) -> TestAccount:
        return TestAccount()

    monkeypatch.setattr(artifacts.pwd, "getpwuid", account_lookup)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    index_path = artifact_root / "artifact-index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    bundle = artifact_root / "policy.safetensors"
    expected = {
        "empty": torch.empty((0,), dtype=torch.float32),
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    save_bundle(bundle, expected)
    config = artifact_root / "resolved.json"
    normalizer = artifact_root / "normalizer.json"
    checkpoint = artifact_root / "checkpoint.ckpt"
    manifest = artifact_root / "manifest.json"
    config.write_text("{}\n", encoding="utf-8")
    normalizer.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    identity = trusted_identity("dp_cnn", "a" * 64, "b" * 64, optimizer_updates=100_000)
    manifest.write_text(json.dumps(identity.bundle_manifest(checkpoint, config)), encoding="utf-8")
    production_receipt = artifact_root / "training_receipt.json"
    production_receipt.write_text(
        json.dumps(
            {
                "schema": "pusht-so100-full-training-v1",
                "model": "dp_cnn",
                "training_mode": "full_production",
                "configured_optimizer_updates": 100_000,
                "rollout_during_training": False,
                "completed": True,
                "identity": identity.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    training_log = artifact_root / "training.log"
    training_log.write_text("training complete\n", encoding="utf-8")
    index = ArtifactIndex(index_path, artifact_root)
    scope = ArtifactScope(
        training_mode="full_production",
        identity=identity.to_dict(),
        training_log=training_log,
    )
    index.anchor_checkpoint(
        "probe",
        checkpoint,
        ArtifactScope(
            config=config,
            training_mode="full_production",
            identity=identity.to_dict(),
            production_receipt=production_receipt,
            training_log=training_log,
        ),
    )
    index.anchor_bundle("probe", BundleFiles(bundle, config, normalizer, manifest), scope)
    expectation = BundleExpectation(identity, sha256_file(checkpoint))
    loaded = load_bundle(
        bundle, expected, index=index, artifact_id="probe", expectation=expectation
    )
    assert torch.equal(loaded["weight"], expected["weight"])
    record = index.record("probe")
    assert record["deployment_scope"] == "simulation_only"
    assert record["training_eligible"] is True
    assert record["bundle_sha256"] == sha256_file(bundle)
    with pytest.raises(ArtifactError, match="keys"):
        load_bundle(
            bundle,
            {"other": expected["weight"]},
            index=index,
            artifact_id="probe",
            expectation=expectation,
        )
    with pytest.raises(ArtifactError, match="shape"):
        load_bundle(
            bundle,
            {"empty": expected["empty"], "weight": torch.empty(3, 2)},
            index=index,
            artifact_id="probe",
            expectation=expectation,
        )


def test_altered_digest_and_unsafe_artifact_paths_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    index_path = tmp_path / "index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    file = root / "checkpoint.ckpt"
    file.write_bytes(b"native")
    index = ArtifactIndex(index_path, root)
    config = root / "resolved.json"
    config.write_text("{}\n", encoding="utf-8")
    index.anchor_checkpoint("run", file, ArtifactScope(config=config, simulation_probe=True))
    file.write_bytes(b"altered")
    with pytest.raises(ArtifactError, match="digest"):
        index.verify("run", "checkpoint")
    config.write_text('{"altered":true}\n', encoding="utf-8")
    with pytest.raises(ArtifactError, match="digest"):
        index.verify("run", "config")
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"x")
    with pytest.raises(ArtifactError, match="artifact root"):
        index.anchor_checkpoint("outside", outside, ArtifactScope(simulation_probe=True))
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(ArtifactError, match="symlink"):
        index.anchor_checkpoint("link", link, ArtifactScope(simulation_probe=True))
    fifo = root / "fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(ArtifactError, match="regular file"):
            index.anchor_checkpoint("fifo", fifo, ArtifactScope(simulation_probe=True))
    finally:
        fifo.unlink()
    unsafe_parent = root / "unsafe-parent"
    unsafe_parent.symlink_to(tmp_path)
    with pytest.raises(ArtifactError, match="symlink"):
        index.create_output_directory(unsafe_parent / "run")
    regular = root / "regular"
    regular.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ArtifactError, match="directory"):
        index.create_output_directory(regular / "run")


def test_evaluator_cli_is_bundle_only(tmp_path: Path) -> None:
    from so101_pusht_benchmark.cli import command_parser

    bundle = tmp_path / "model.safetensors"
    parser = command_parser()
    args = parser.parse_args(["evaluate-model", "--bundle", str(bundle)])
    assert args.bundle == bundle
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-model", "--bundle", str(bundle), "--checkpoint", "x"])


def test_index_json_remains_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = tmp_path / "index.json"
    path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    item = root / "x"
    item.write_bytes(b"x")
    ArtifactIndex(path, root).anchor_checkpoint("x", item, ArtifactScope(simulation_probe=True))
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert list(parsed) == ["schema", "artifacts"]


def test_workspace_config_honors_lstm_batch_size_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PUSHT_LSTM_BATCH_SIZE", "8")

    config = workspace_config("lstm_gmm", tmp_path / "paper", 0)
    diffusion_config = workspace_config("dp_cnn", tmp_path / "paper", 0)

    assert config["dataloader"]["batch_size"] == 8
    assert config["val_dataloader"]["batch_size"] == 8
    assert diffusion_config["dataloader"]["batch_size"] == 32
