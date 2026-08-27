from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import cast

import pytest
import torch

from so101_pusht_benchmark.integrations.paper_baselines.configs import PROFILES, workspace_config
from so101_pusht_benchmark.training.artifacts import (
    ArtifactError,
    ArtifactIndex,
    ArtifactScope,
    BundleFiles,
    require_production_artifact,
    sha256_file,
)
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle, save_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity, trusted_identity
from so101_pusht_benchmark.training.metadata import (
    read_normalizer_metadata,
    read_trusted_config,
)
from so101_pusht_benchmark.training import validate_production_resume_artifact


def _write_parallel_record(index_path: str, artifact_root: str, artifact_id: str) -> None:
    index = ArtifactIndex(Path(index_path), Path(artifact_root))
    index.merge_record(artifact_id, {"model": artifact_id})


def test_artifact_index_parallel_writers_preserve_every_record(tmp_path: Path) -> None:
    index_path = tmp_path / "artifact-index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_write_parallel_record,
            args=(str(index_path), str(tmp_path), f"model-{index}"),
        )
        for index in range(12)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    assert all(process.exitcode == 0 for process in processes)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(document["artifacts"]) == {f"model-{index}" for index in range(12)}


@pytest.fixture(autouse=True)
def secure_receipt_home(secure_test_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from so101_pusht_benchmark.training import artifacts

    class TestAccount:
        pw_dir = str(secure_test_home)

    def account_lookup(_uid: int) -> TestAccount:
        return TestAccount()

    monkeypatch.setattr(artifacts.pwd, "getpwuid", account_lookup)


def _index(tmp_path: Path) -> tuple[ArtifactIndex, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    path = root / "artifact-index.json"
    path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    return ArtifactIndex(path, root), root


def _anchor_full_production(
    index: ArtifactIndex, root: Path, artifact_id: str = "dp-cnn-production"
) -> tuple[Path, Path]:
    output = root / "models/dp_cnn/full"
    checkpoint = output / "checkpoints/latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted full production")
    config = output / "resolved_config.json"
    config.write_text(
        json.dumps(workspace_config("dp_cnn", "/trusted/frozen-view", 0)),
        encoding="utf-8",
    )
    identity = trusted_identity(
        "dp_cnn", "a" * 64, "b" * 64, optimizer_updates=1_794_000
    ).to_dict()
    receipt = output / "training_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "pusht-so100-full-training-v1",
                "model": "dp_cnn",
                "training_mode": "full_production",
                "configured_optimizer_updates": 1_794_000,
                "executed_optimizer_updates": 1_794_000,
                "rollout_during_training": False,
                "completed": True,
                "identity": identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    training_log = output / "logs.json.txt"
    training_log.write_text('{"loss":0.1}\n', encoding="utf-8")
    index.anchor_checkpoint(
        artifact_id,
        checkpoint,
        ArtifactScope(
            config=config,
            training_mode="full_production",
            identity=identity,
            production_receipt=receipt,
            training_log=training_log,
        ),
    )
    return checkpoint, config


@pytest.mark.parametrize("tamper", ["bytes", "status", "foreign", "unexpected"])
def test_training_resume_validation_fails_closed_without_mutation(
    tmp_path: Path, tamper: str
) -> None:
    index, root = _index(tmp_path)
    checkpoint, _ = _anchor_full_production(index, root)
    output = root / "models/dp_cnn/full"
    if tamper == "bytes":
        checkpoint.chmod(0o600)
        checkpoint.write_bytes(b"tampered")
    elif tamper == "status":
        raw = json.loads(index.path.read_text(encoding="utf-8"))
        raw["artifacts"]["dp-cnn-production"]["result_status"] = "candidate"
        index.path.write_text(json.dumps(raw), encoding="utf-8")
    elif tamper == "unexpected":
        (output / "foreign.txt").write_bytes(b"keep")
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    if tamper == "foreign":
        with pytest.raises(ArtifactError, match="artifact ID"):
            validate_production_resume_artifact(
                index,
                stage="training",
                model="ibc",
                artifact_id="dp-cnn-production",
                output=output,
            )
    else:
        with pytest.raises(ArtifactError):
            validate_production_resume_artifact(
                index,
                stage="training",
                model="dp_cnn",
                artifact_id="dp-cnn-production",
                output=output,
            )
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == before


def test_valid_training_and_bundle_resume_are_read_only(tmp_path: Path) -> None:
    index, root = _index(tmp_path)
    checkpoint, config = _anchor_full_production(index, root)
    training = root / "models/dp_cnn/full"
    assert (
        validate_production_resume_artifact(
            index,
            stage="training",
            model="dp_cnn",
            artifact_id="dp-cnn-production",
            output=training,
        )["status"]
        == "validated"
    )

    bundle_root = root / "models/dp_cnn/bundle"
    bundle_root.mkdir(parents=True)
    bundle = bundle_root / "policy.safetensors"
    bundle.write_bytes(b"tensor-only fixture")
    bundle_config = bundle_root / "resolved_config.json"
    bundle_config.write_bytes(config.read_bytes())
    identity = BundleIdentity.from_dict(index.record("dp-cnn-production")["identity"])
    normalizer = bundle_root / "normalizer.json"
    normalizer.write_text(
        json.dumps(
            {
                "schema": 1,
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "resolved_config_sha256": sha256_file(bundle_config),
                "identity": identity.to_dict(),
                "state": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = bundle_root / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(identity.bundle_manifest(checkpoint, bundle_config), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    index.anchor_bundle(
        "dp-cnn-production",
        BundleFiles(bundle, bundle_config, normalizer, manifest),
        ArtifactScope(training_mode="full_production", identity=identity.to_dict()),
    )
    before = {path.name: path.read_bytes() for path in bundle_root.iterdir()}

    assert (
        validate_production_resume_artifact(
            index,
            stage="bundle",
            model="dp_cnn",
            artifact_id="dp-cnn-production",
            output=bundle_root,
        )["status"]
        == "validated"
    )
    assert {path.name: path.read_bytes() for path in bundle_root.iterdir()} == before


@pytest.mark.parametrize("tamper", ["bundle-bytes", "manifest", "unexpected", "foreign"])
def test_bundle_resume_tamper_fails_closed(tmp_path: Path, tamper: str) -> None:
    index, root = _index(tmp_path)
    checkpoint, config = _anchor_full_production(index, root)
    bundle_root = root / "models/dp_cnn/bundle"
    bundle_root.mkdir(parents=True)
    bundle = bundle_root / "policy.safetensors"
    bundle.write_bytes(b"bundle")
    bundle_config = bundle_root / "resolved_config.json"
    bundle_config.write_bytes(config.read_bytes())
    identity = BundleIdentity.from_dict(index.record("dp-cnn-production")["identity"])
    normalizer = bundle_root / "normalizer.json"
    normalizer.write_text(
        json.dumps(
            {
                "schema": 1,
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "resolved_config_sha256": sha256_file(bundle_config),
                "identity": identity.to_dict(),
                "state": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = bundle_root / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(identity.bundle_manifest(checkpoint, bundle_config)), encoding="utf-8"
    )
    index.anchor_bundle(
        "dp-cnn-production",
        BundleFiles(bundle, bundle_config, normalizer, manifest),
        ArtifactScope(training_mode="full_production", identity=identity.to_dict()),
    )
    if tamper == "bundle-bytes":
        bundle.chmod(0o600)
        bundle.write_bytes(b"tampered")
    elif tamper == "manifest":
        manifest.chmod(0o600)
        manifest.write_text("{}", encoding="utf-8")
    elif tamper == "unexpected":
        (bundle_root / "foreign").write_bytes(b"keep")
    before = {path.name: path.read_bytes() for path in bundle_root.iterdir()}

    with pytest.raises(ArtifactError):
        validate_production_resume_artifact(
            index,
            stage="bundle",
            model="ibc" if tamper == "foreign" else "dp_cnn",
            artifact_id="dp-cnn-production",
            output=bundle_root,
        )
    assert {path.name: path.read_bytes() for path in bundle_root.iterdir()} == before


@pytest.mark.parametrize(
    "tamper",
    [
        "forged-index",
        "receipt-bytes",
        "receipt-mode",
        "receipt-symlink",
        "extra-receipt",
        "ownership",
    ],
)
def test_bundle_resume_requires_immutable_authenticated_stage_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    from so101_pusht_benchmark.training import artifacts

    index, root = _index(tmp_path)
    checkpoint, config = _anchor_full_production(index, root)
    output = root / "models/dp_cnn/bundle"
    output.mkdir(parents=True)
    bundle = output / "policy.safetensors"
    bundle.write_bytes(b"bundle")
    bundle_config = output / "resolved_config.json"
    bundle_config.write_bytes(config.read_bytes())
    identity = BundleIdentity.from_dict(index.record("dp-cnn-production")["identity"])
    normalizer = output / "normalizer.json"
    normalizer.write_text(
        json.dumps(
            {
                "schema": 1,
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "resolved_config_sha256": sha256_file(bundle_config),
                "identity": identity.to_dict(),
                "state": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = output / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(identity.bundle_manifest(checkpoint, bundle_config)), encoding="utf-8"
    )
    index.anchor_bundle(
        "dp-cnn-production",
        BundleFiles(bundle, bundle_config, normalizer, manifest),
        ArtifactScope(training_mode="full_production", identity=identity.to_dict()),
    )
    receipt = index.stage_receipt_path("dp-cnn-production", "bundle")
    if tamper == "forged-index":
        bundle.chmod(0o600)
        bundle.write_bytes(b"attacker replacement")
        bundle.chmod(0o400)
        raw = json.loads(index.path.read_text(encoding="utf-8"))
        raw["artifacts"]["dp-cnn-production"]["bundle_sha256"] = sha256_file(bundle)
        index.path.write_text(json.dumps(raw), encoding="utf-8")
    elif tamper == "receipt-bytes":
        receipt.chmod(0o600)
        receipt.write_text("{}\n", encoding="utf-8")
        receipt.chmod(0o400)
    elif tamper == "receipt-mode":
        receipt.chmod(0o600)
    elif tamper == "receipt-symlink":
        foreign = tmp_path / "foreign-receipt.json"
        foreign.write_bytes(receipt.read_bytes())
        foreign.chmod(0o400)
        receipt.unlink()
        receipt.symlink_to(foreign)
    elif tamper == "extra-receipt":
        extra = index.receipt_root / "extra.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o400)
    else:
        actual_uid = artifacts.os.getuid()
        monkeypatch.setattr(artifacts.os, "getuid", lambda: actual_uid + 1)
    before = {
        path.name: (path.readlink().as_posix() if path.is_symlink() else path.read_bytes())
        for path in output.iterdir()
    }

    with pytest.raises(ArtifactError):
        validate_production_resume_artifact(
            index,
            stage="bundle",
            model="dp_cnn",
            artifact_id="dp-cnn-production",
            output=output,
        )

    assert {
        path.name: (path.readlink().as_posix() if path.is_symlink() else path.read_bytes())
        for path in output.iterdir()
    } == before


@pytest.mark.parametrize("model", tuple(PROFILES))
def test_trusted_config_accepts_all_four_profiles(tmp_path: Path, model: str) -> None:
    config = workspace_config(model, "/trusted/frozen-view", 0)
    path = tmp_path / f"{model}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert read_trusted_config(path, model)["name"] == model
    with pytest.raises(ArtifactError, match="identity"):
        read_trusted_config(path, next(name for name in PROFILES if name != model))


def test_fixture_artifact_is_rejected_without_status_mutation() -> None:
    fixture = {
        "training_eligible": False,
        "comparison_eligible": False,
        "result_status": "ineligible_fixture",
    }
    before = dict(fixture)
    with pytest.raises(ArtifactError, match="production"):
        require_production_artifact(fixture, operation="export")
    assert fixture == before


def test_production_smoke_is_nonfinal_and_rejected_by_export_and_evaluation() -> None:
    smoke = {
        "deployment_scope": "simulation_only",
        "training_eligible": False,
        "comparison_eligible": False,
        "result_status": "production_smoke_complete_nonfinal",
        "identity": trusted_identity("dp_cnn", "a" * 64, "b" * 64).to_dict(),
    }
    for operation in ("bundle export", "evaluation"):
        with pytest.raises(ArtifactError, match="smoke checkpoints are non-final"):
            require_production_artifact(smoke, operation=operation)


def test_production_smoke_export_rejects_before_runtime_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_pusht_benchmark.training import exporter

    index, root = _index(tmp_path)
    checkpoint = root / "smoke.ckpt"
    checkpoint.write_bytes(b"non-final smoke")
    config = root / "smoke.json"
    config.write_text("{}\n", encoding="utf-8")
    identity = trusted_identity("dp_cnn", "a" * 64, "b" * 64)
    index.anchor_checkpoint(
        "dp-cnn-production-smoke",
        checkpoint,
        ArtifactScope(config=config, smoke_mode="production", identity=identity.to_dict()),
    )
    monkeypatch.setattr(
        exporter,
        "assert_paper_runtime",
        lambda: pytest.fail("smoke rejection must precede runtime and checkpoint reload"),
    )
    output = root / "must-not-exist"
    before = index.path.read_bytes()
    with pytest.raises(ArtifactError, match="smoke checkpoints are non-final"):
        exporter.export_inference_bundle(
            checkpoint,
            config,
            output,
            artifact_id="dp-cnn-production-smoke",
            index=index,
        )
    assert index.path.read_bytes() == before
    assert not output.exists()


def test_forged_full_production_index_rejects_before_runtime_or_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_pusht_benchmark.training import exporter

    index, root = _index(tmp_path)
    checkpoint = root / "evil.ckpt"
    checkpoint.write_bytes(b"attacker selected pickle")
    config = root / "evil.json"
    config.write_text("{}\n", encoding="utf-8")
    identity = trusted_identity(
        "dp_cnn", "a" * 64, "b" * 64, optimizer_updates=1_794_000
    ).to_dict()
    forged = {
        "deployment_scope": "simulation_only",
        "training_eligible": True,
        "comparison_eligible": False,
        "result_status": "full_training_complete",
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": config.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config),
        "identity": identity,
    }
    index.path.write_text(
        json.dumps({"schema": 1, "artifacts": {"evil": forged}}), encoding="utf-8"
    )
    calls = {"runtime": 0, "workspace": 0, "unsafe_import": 0}
    original_import = __import__

    def import_counter(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith(("diffusion_policy", "omegaconf")):
            calls["unsafe_import"] += 1
        return original_import(name, globals, locals, fromlist, level)

    def runtime_counter() -> None:
        calls["runtime"] += 1

    def workspace_counter(_model: str) -> type[object]:
        calls["workspace"] += 1
        return object

    monkeypatch.setattr("builtins.__import__", import_counter)
    monkeypatch.setattr(exporter, "assert_paper_runtime", runtime_counter)
    monkeypatch.setattr(exporter, "resolve_workspace_class", workspace_counter)
    with pytest.raises(ArtifactError, match="producer receipt"):
        exporter.export_inference_bundle(
            checkpoint,
            config,
            root / "must-not-exist",
            artifact_id="evil",
            index=index,
        )
    assert calls == {"runtime": 0, "workspace": 0, "unsafe_import": 0}


def test_forged_index_cli_exits_nonzero_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_pusht_benchmark import native_cli
    from so101_pusht_benchmark.training import exporter

    index, root = _index(tmp_path)
    checkpoint = root / "evil.ckpt"
    checkpoint.write_bytes(b"executable pickle")
    config = root / "evil.json"
    config.write_text("{}\n", encoding="utf-8")
    index.path.write_text(
        json.dumps(
            {
                "schema": 1,
                "artifacts": {
                    "evil": {
                        "deployment_scope": "simulation_only",
                        "training_eligible": True,
                        "comparison_eligible": False,
                        "result_status": "full_training_complete",
                        "checkpoint_path": "evil.ckpt",
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "config_path": "evil.json",
                        "config_sha256": sha256_file(config),
                        "identity": trusted_identity(
                            "dp_cnn", "a" * 64, "b" * 64, optimizer_updates=1_794_000
                        ).to_dict(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls = {"runtime": 0, "workspace": 0}

    def count_workspace(_model: str) -> type[object]:
        calls["workspace"] = 1
        return object

    monkeypatch.setattr(exporter, "assert_paper_runtime", lambda: calls.__setitem__("runtime", 1))
    monkeypatch.setattr(exporter, "resolve_workspace_class", count_workspace)
    result = native_cli.main(
        [
            "export-inference-bundle",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--output",
            str(root / "output"),
            "--artifact-id",
            "evil",
            "--artifact-index",
            str(index.path),
        ]
    )
    assert result != 0
    assert calls == {"runtime": 0, "workspace": 0}


@pytest.mark.parametrize("mutation", ["index", "checkpoint", "receipt", "duplicate"])
def test_tampered_or_duplicated_production_chain_rejects_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from so101_pusht_benchmark.training import exporter

    index, root = _index(tmp_path)
    checkpoint, config = _anchor_full_production(index, root)
    if mutation == "index":
        raw = json.loads(index.path.read_text(encoding="utf-8"))
        raw["artifacts"]["dp-cnn-production"]["training_eligible"] = False
        index.path.write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "checkpoint":
        checkpoint.chmod(0o600)
        checkpoint.write_bytes(b"tampered pickle")
    elif mutation == "receipt":
        receipt = next(index.receipt_root.glob("*.json"))
        receipt.chmod(0o600)
        receipt.write_text("{}\n", encoding="utf-8")
    else:
        receipt = next(index.receipt_root.glob("*.json"))
        duplicate = receipt.with_name("duplicate.json")
        duplicate.write_bytes(receipt.read_bytes())
        duplicate.chmod(0o400)
    calls = {"runtime": 0, "workspace": 0}

    def count_workspace(_model: str) -> type[object]:
        calls["workspace"] = 1
        return object

    monkeypatch.setattr(exporter, "assert_paper_runtime", lambda: calls.__setitem__("runtime", 1))
    monkeypatch.setattr(exporter, "resolve_workspace_class", count_workspace)
    with pytest.raises(ArtifactError):
        exporter.export_inference_bundle(
            checkpoint,
            config,
            root / "must-not-exist",
            artifact_id="dp-cnn-production",
            index=index,
        )
    assert calls == {"runtime": 0, "workspace": 0}


def test_authenticated_full_production_checkpoint_exports_after_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy

    from so101_pusht_benchmark.training import exporter, model_smoke

    index, root = _index(tmp_path)
    checkpoint, config = _anchor_full_production(index, root)
    calls: list[Path] = []

    class FakePolicy(BaseImagePolicy):
        pass

    class FakeWorkspace:
        def __init__(self, _config: object, *, output_dir: str) -> None:
            del output_dir
            self.model = FakePolicy()

        def load_checkpoint(self, *, path: Path) -> None:
            calls.append(path)

    def workspace_class(_model: str) -> type[FakeWorkspace]:
        return FakeWorkspace

    def accept_identity(_model: str, _policy: object) -> None:
        return None

    monkeypatch.setattr(exporter, "assert_paper_runtime", lambda: None)
    monkeypatch.setattr(exporter, "resolve_workspace_class", workspace_class)
    monkeypatch.setattr(model_smoke, "validate_model_identity", accept_identity)
    output = root / "models/dp_cnn/bundle"
    bundle = exporter.export_inference_bundle(
        checkpoint,
        config,
        output,
        artifact_id="dp-cnn-production",
        index=index,
    )
    assert calls == [checkpoint]
    assert bundle == output / "policy.safetensors"
    assert bundle.is_file()
    assert index.record("dp-cnn-production")["result_status"] == "full_training_bundle_ready"


def test_fixture_checkpoint_export_rejects_before_reload_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101_pusht_benchmark.training import exporter

    index, root = _index(tmp_path)
    index.path.write_text(
        json.dumps(
            {
                "schema": 1,
                "artifacts": {
                    "fixture": {
                        "deployment_scope": "simulation_only",
                        "training_eligible": False,
                        "comparison_eligible": False,
                        "result_status": "ineligible_fixture",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exporter,
        "assert_paper_runtime",
        lambda: pytest.fail("fixture rejection must precede checkpoint runtime/reload"),
    )
    before = index.path.read_bytes()
    output = root / "bundle-output"
    with pytest.raises(ArtifactError, match="production"):
        exporter.export_inference_bundle(
            root / "missing.ckpt",
            root / "missing.json",
            output,
            artifact_id="fixture",
            index=index,
        )
    assert index.path.read_bytes() == before
    assert not output.exists()


def test_bundle_identity_binds_every_manifest_and_rejects_mix_or_tamper(tmp_path: Path) -> None:
    index, root = _index(tmp_path)
    checkpoint = root / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = root / "resolved_config.json"
    config.write_text("{}\n", encoding="utf-8")
    identity = trusted_identity("ibc", "a" * 64, "b" * 64, optimizer_updates=100_000)
    production_receipt = root / "training_receipt.json"
    production_receipt.write_text(
        json.dumps(
            {
                "schema": "pusht-so100-full-training-v1",
                "model": "ibc",
                "training_mode": "full_production",
                "configured_optimizer_updates": 100_000,
                "executed_optimizer_updates": 100_000,
                "rollout_during_training": False,
                "completed": True,
                "identity": identity.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    training_log = root / "logs.json.txt"
    training_log.write_text('{"loss":0.1}\n', encoding="utf-8")
    index.anchor_checkpoint(
        "ibc",
        checkpoint,
        ArtifactScope(
            config=config,
            training_mode="full_production",
            identity=identity.to_dict(),
            production_receipt=production_receipt,
            training_log=training_log,
        ),
    )

    bundle = root / "policy.safetensors"
    expected = {"weight": torch.ones(2, dtype=torch.float32)}
    save_bundle(bundle, expected)
    normalizer = root / "normalizer.json"
    normalizer.write_text(
        '{"schema":1,"state":{"normalizer.x":{"shape":[2],"dtype":"torch.float32"}}}\n',
        encoding="utf-8",
    )
    manifest = root / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(identity.bundle_manifest(checkpoint, config), sort_keys=True), encoding="utf-8"
    )
    index.anchor_bundle(
        "ibc",
        BundleFiles(bundle, config, normalizer, manifest),
        ArtifactScope(training_mode="full_production", identity=identity.to_dict()),
    )
    require_production_artifact(index.record("ibc"), operation="evaluation")
    checkpoint_digest = sha256_file(checkpoint)
    loaded = load_bundle(
        bundle,
        expected,
        index=index,
        artifact_id="ibc",
        expectation=BundleExpectation(identity, checkpoint_digest),
    )
    assert torch.equal(loaded["weight"], expected["weight"])

    mixed = trusted_identity("dp_cnn", "a" * 64, "b" * 64)
    with pytest.raises(ArtifactError, match="identity"):
        load_bundle(
            bundle,
            expected,
            index=index,
            artifact_id="ibc",
            expectation=BundleExpectation(mixed, checkpoint_digest),
        )
    raw = bytearray(bundle.read_bytes())
    raw[-1] ^= 1
    bundle.chmod(0o600)
    bundle.write_bytes(raw)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        load_bundle(
            bundle,
            expected,
            index=index,
            artifact_id="ibc",
            expectation=BundleExpectation(identity, checkpoint_digest),
        )


def test_unknown_model_origin_is_rejected() -> None:
    with pytest.raises(ArtifactError, match="unknown model"):
        trusted_identity("local_model", "a" * 64, "b" * 64)
    with pytest.raises(ArtifactError, match="SHA-256"):
        BundleIdentity.from_dict({"model": "dp_cnn"})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unknown",), True),
        (("policy", "unknown"), True),
        (("task", "dataset", "unknown"), True),
        (("optimizer", "unknown"), True),
    ],
)
def test_trusted_config_rejects_unknown_keys_recursively(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    config = workspace_config("dp_cnn", "/trusted/frozen-view", 0)
    target = config
    for key in path[:-1]:
        child: object = target[key]
        assert isinstance(child, dict)
        target = cast("dict[str, object]", child)
    target[path[-1]] = value
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ArtifactError, match="keys"):
        read_trusted_config(config_path, "dp_cnn")


def test_normalizer_metadata_is_an_exact_recursive_allowlist(tmp_path: Path) -> None:
    identity = trusted_identity("ibc", "a" * 64, "b" * 64)
    valid = {
        "schema": 1,
        "deployment_scope": "simulation_only",
        "training_eligible": True,
        "source_checkpoint_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
        "identity": identity.to_dict(),
        "state": {"normalizer.x": {"shape": [2], "dtype": "torch.float32"}},
    }
    path = tmp_path / "normalizer.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert read_normalizer_metadata(path, identity, "c" * 64, "d" * 64) == {
        "normalizer.x": ((2,), "torch.float32")
    }
    mixed = {**valid, "identity": trusted_identity("dp_cnn", "a" * 64, "b" * 64).to_dict()}
    path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(ArtifactError, match="trusted identity mismatch"):
        read_normalizer_metadata(path, identity, "c" * 64, "d" * 64)

    for mutation in (
        {**valid, "unknown": True},
        {**valid, "state": {"normalizer.x": {"shape": [2], "dtype": "torch.float32", "x": 1}}},
        {key: value for key, value in valid.items() if key != "identity"},
    ):
        path.write_text(json.dumps(mutation), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"schema|fields|tensor"):
            read_normalizer_metadata(path, identity, "c" * 64, "d" * 64)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("observation_steps", True),
        ("horizon", 2.0),
        ("executed_actions", False),
        ("model", 7),
        ("stanford_commit", True),
    ],
)
def test_bundle_identity_rejects_coercive_runtime_types(field: str, bad: object) -> None:
    raw = trusted_identity("ibc", "a" * 64, "b" * 64).to_dict()
    raw[field] = bad
    with pytest.raises(ArtifactError, match=r"type|identity|integer|string"):
        BundleIdentity.from_dict(raw)
