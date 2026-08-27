from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import cast

import numpy as np
import pytest

from so101_pusht_benchmark.data.paper_view import LoadedPaperView
from so101_pusht_benchmark.evaluation.frozen_env import ActionContractError, FrozenStep
from so101_pusht_benchmark.native_runtime import NativeRuntimeReport
from so101_pusht_benchmark.training.artifacts import ArtifactIndex
from so101_pusht_benchmark.training.vertical_smoke import (
    MODEL_SMOKE_SCHEMA,
    ModelSmokeError,
    ReloadValidationContext,
    SmokeDependencies,
    SmokeIdentityError,
    load_inference_action,
    run_model_smoke,
    validate_model_smoke_result,
    validate_reload_receipt,
)
from so101_pusht_benchmark.workspace import runtime_artifact_root


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_split(dataset_digest: str) -> str:
    return hashlib.sha256(f"ineligible-fixture:{dataset_digest}".encode()).hexdigest()


def _identity(model: str, *, dataset_digest: str = "a" * 64) -> dict[str, object]:
    profiles = {
        "dp_cnn": (
            "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy",
            "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace",
            2,
            16,
            8,
        ),
        "ibc": (
            "diffusion_policy.policy.ibc_dfo_hybrid_image_policy.IbcDfoHybridImagePolicy",
            "diffusion_policy.workspace.train_ibc_dfo_hybrid_workspace.TrainIbcDfoHybridWorkspace",
            2,
            2,
            1,
        ),
    }
    policy, workspace, observation_steps, horizon, executed_actions = profiles[model]
    return {
        "model": model,
        "policy_target": policy,
        "workspace_target": workspace,
        "observation_steps": observation_steps,
        "horizon": horizon,
        "executed_actions": executed_actions,
        "optimizer_updates": 1,
        "dataset_digest": dataset_digest,
        "split_digest": _fixture_split(dataset_digest),
        "runtime_lock_digest": "c" * 64,
        "environment_manifest_digest": "d" * 64,
        "stanford_commit": "5ba07ac6661db573af695b419a7947ecb704690f",
        "robomimic_commit": "62ed2de905caeb9133136e4d14d810a8b6baa96c",
    }


def _view() -> LoadedPaperView:
    arrays = {
        "cam_top": np.zeros((16, 224, 224, 3), dtype=np.uint8),
        "cam_side": np.ones((16, 224, 224, 3), dtype=np.uint8),
        "agent_pos": np.zeros((16, 5), dtype=np.float32),
    }
    return LoadedPaperView(
        arrays=cast("dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]", arrays),
        episode_ends=np.asarray([16], dtype=np.int64),
        manifest={
            "canonical_digest": "a" * 64,
            "root_digest": "f" * 64,
            "training_eligible": False,
        },
        splits={
            "frozen": False,
            "training_eligible": False,
            "reason": "synthetic_fixture_not_comparison_eligible",
        },
    )


def _receipt(
    root: Path, model: str = "dp_cnn"
) -> tuple[Path, dict[str, object], ArtifactIndex, Path]:
    store = root / "fixture-store"
    store.mkdir(exist_ok=True)
    manifest = store / "manifest.json"
    manifest.write_text('{"canonical_digest":"fixture"}\n', encoding="utf-8")
    splits = store / "splits.json"
    splits.write_text('{"reason":"synthetic"}\n', encoding="utf-8")
    checkpoint = root / "checkpoint.ckpt"
    checkpoint.write_bytes(b"strict checkpoint")
    config = root / "resolved_config.json"
    config.write_text(
        json.dumps(
            {
                "name": model,
                "_target_": _identity(model)["workspace_target"],
                "policy": {"_target_": _identity(model)["policy_target"]},
                "task": {"dataset": {"zarr_path": str(store)}},
            }
        ),
        encoding="utf-8",
    )
    identity = _identity(model)
    receipt: dict[str, object] = {
        "schema": MODEL_SMOKE_SCHEMA,
        "phase": "strict_reload",
        "model": model,
        "fixture": True,
        "production_eligible": False,
        "comparison_eligible": False,
        "identity": identity,
        "store_identity": {
            "canonical_digest": "a" * 64,
            "root_digest": "f" * 64,
            "split_digest": _fixture_split("a" * 64),
            "manifest_sha256": _digest(manifest),
            "splits_sha256": _digest(splits),
        },
        "checkpoint": "checkpoint.ckpt",
        "checkpoint_sha256": _digest(checkpoint),
        "config": "resolved_config.json",
        "config_sha256": _digest(config),
        "reload_verified": True,
        "policy_class": str(identity["policy_target"]).rsplit(".", 1)[1],
        "optimizer_updates": 1,
        "loss": 1.25,
    }
    path = root / "reload.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    index_path = root / "artifact-index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "artifacts": {
                    "fixture-model-smoke": {
                        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                        "checkpoint_sha256": _digest(checkpoint),
                        "config_path": config.relative_to(root).as_posix(),
                        "config_sha256": _digest(config),
                        "identity": identity,
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
    return path, receipt, ArtifactIndex(index_path, root), store


def _validate(path: Path, index: ArtifactIndex, store: Path) -> dict[str, object]:
    return validate_reload_receipt(
        path,
        expected_model="dp_cnn",
        context=ReloadValidationContext(
            index.artifact_root,
            index,
            "fixture-model-smoke",
            store,
            _view(),
        ),
    )


def _write_action(root: Path, value: np.ndarray[tuple[int, ...], np.dtype[np.float32]]) -> None:
    value.tofile(root / "action.bin")
    (root / "inference.json").write_text(
        json.dumps(
            {
                "schema": MODEL_SMOKE_SCHEMA,
                "model": "dp_cnn",
                "checkpoint_reloaded": True,
                "action_dtype": "float32",
                "action_shape": list(value.shape),
                "action": value.tolist(),
            }
        ),
        encoding="utf-8",
    )


def test_reload_receipt_binds_model_config_checkpoint_and_store_before_environment(
    tmp_path: Path,
) -> None:
    path, receipt, index, store = _receipt(tmp_path)
    validated = _validate(path, index, store)
    assert validated["model"] == "dp_cnn"

    mutations: list[tuple[str, dict[str, object]]] = []
    wrong_model = {**receipt, "model": "ibc", "identity": _identity("ibc")}
    mutations.append(("model", wrong_model))
    wrong_identity = json.loads(json.dumps(receipt))
    cast("dict[str, object]", wrong_identity["identity"])["policy_target"] = _identity("ibc")[
        "policy_target"
    ]
    mutations.append(("identity", wrong_identity))
    forged_dataset = json.loads(json.dumps(receipt))
    cast("dict[str, object]", forged_dataset["identity"])["dataset_digest"] = "0" * 64
    mutations.append(("dataset", forged_dataset))
    mutations.append(("config", {**receipt, "config_sha256": "0" * 64}))
    mutations.append(("checkpoint", {**receipt, "checkpoint_sha256": "0" * 64}))

    for label, mutation in mutations:
        path.write_text(json.dumps(mutation), encoding="utf-8")
        with pytest.raises(SmokeIdentityError, match=label):
            _validate(path, index, store)


def test_deliberate_dp_cnn_as_ibc_reload_is_rejected(tmp_path: Path) -> None:
    path, _, index, store = _receipt(tmp_path, "dp_cnn")
    with pytest.raises(SmokeIdentityError, match="model"):
        validate_reload_receipt(
            path,
            expected_model="ibc",
            context=ReloadValidationContext(
                tmp_path,
                index,
                "fixture-model-smoke",
                store,
                _view(),
            ),
        )


def test_inference_action_is_strictly_loaded_before_environment(tmp_path: Path) -> None:
    _write_action(tmp_path, np.asarray([0.25, -0.25], dtype=np.float32))
    action = load_inference_action(tmp_path, expected_model="dp_cnn")
    assert action.dtype == np.dtype(np.float32)
    assert action.tolist() == [0.25, -0.25]


@pytest.mark.parametrize(
    ("action", "forge_dataset", "expected_error"),
    [
        (np.asarray([0.0, 0.0], dtype=np.float32), True, SmokeIdentityError),
        (np.asarray([float("nan"), 0.0], dtype=np.float32), False, ActionContractError),
        (np.asarray([0.0, 0.0, 0.0], dtype=np.float32), False, ModelSmokeError),
        (np.asarray([2.0, 0.0], dtype=np.float32), False, ActionContractError),
    ],
)
def test_forged_digest_and_invalid_action_never_construct_environment_or_publish(
    action: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    forge_dataset: bool,
    expected_error: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    canonical_test_root: Path,
) -> None:
    from so101_pusht_benchmark.training import vertical_smoke

    label = (
        "dataset"
        if forge_dataset
        else "shape"
        if action.shape != (2,)
        else "nonfinite"
        if not np.isfinite(action).all()
        else "range"
    )
    namespace = canonical_test_root.name
    output = runtime_artifact_root() / f"{namespace}-pre-env-{label}"
    fake_python = runtime_artifact_root() / f"{namespace}-fake-python-{label}"
    fake_python.write_text("fixture", encoding="utf-8")
    created = 0

    def fixture(root: Path, *, frames: int = 16) -> Path:
        assert frames == 16
        root.mkdir()
        return root

    def imported(_source: Path, destination: Path) -> int:
        destination.mkdir()
        (destination / "manifest.json").write_text(
            '{"canonical_digest":"fixture"}\n', encoding="utf-8"
        )
        (destination / "splits.json").write_text('{"reason":"synthetic"}\n', encoding="utf-8")
        return 0

    def worker(_command: list[str], *, cwd: Path, environment: object) -> None:
        del environment
        path, receipt, _, _ = _receipt(cwd)
        # The full-run forged-digest regression exercises the parent trust binding.
        if forge_dataset:
            forged = cast("dict[str, object]", receipt["identity"])
            forged["dataset_digest"] = "0" * 64
            path.write_text(json.dumps(receipt), encoding="utf-8")
        _write_action(cwd, action)

    class NeverEnvironment:
        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]], dict[str, object]]:
            del seed
            raise AssertionError("environment reset must not run")

        def step(self, action: object) -> FrozenStep:
            del action
            raise AssertionError("environment step must not run")

        def close(self) -> None:
            return None

    def environment_factory() -> NeverEnvironment:
        nonlocal created
        created += 1
        raise AssertionError("environment must not be constructed")

    runtime = cast(
        "NativeRuntimeReport",
        {
            "status": "compatible",
            "plan": "plan",
            "contract_schema": "pusht-so100-native-v1",
            "lock": "environments/sim-runtime.lock",
            "lock_sha256": "c" * 64,
            "source_environment_sha256": "e" * 64,
            "fallback": "forbidden",
            "runtime": {},
        },
    )

    def loaded(_path: Path) -> LoadedPaperView:
        return _view()

    def runtime_report() -> NativeRuntimeReport:
        return runtime

    monkeypatch.setattr(vertical_smoke, "create_nonproduction_native_fixture", fixture)
    monkeypatch.setattr(vertical_smoke, "import_repo_store", imported)
    monkeypatch.setattr(vertical_smoke, "load_paper_view", loaded)
    monkeypatch.setattr(vertical_smoke, "_run_worker", worker)
    try:
        with pytest.raises(expected_error, match=r"dataset|action"):
            run_model_smoke(
                "dp_cnn",
                output,
                fixture=True,
                dependencies=SmokeDependencies(
                    paper_python=fake_python,
                    runtime_report=runtime_report,
                    environment_factory=environment_factory,
                ),
            )
        assert created == 0
        assert not output.exists()
        assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    finally:
        fake_python.unlink(missing_ok=True)
        shutil.rmtree(output, ignore_errors=True)


def test_vertical_smoke_cancellation_removes_owned_staging_before_environment(
    monkeypatch: pytest.MonkeyPatch, canonical_test_root: Path
) -> None:
    from so101_pusht_benchmark.training import vertical_smoke

    namespace = canonical_test_root.name
    output = runtime_artifact_root() / f"{namespace}-pre-env-cancellation"
    fake_python = runtime_artifact_root() / f"{namespace}-fake-python-cancellation"
    fake_python.write_text("fixture", encoding="utf-8")
    created = 0

    def fixture(root: Path, *, frames: int = 16) -> Path:
        assert frames == 16
        root.mkdir()
        return root

    def imported(_source: Path, destination: Path) -> int:
        destination.mkdir()
        return 0

    def cancelled(_command: list[str], *, cwd: Path, environment: object) -> None:
        del cwd, environment
        raise KeyboardInterrupt("injected cancellation")

    def loaded(_path: Path) -> LoadedPaperView:
        return _view()

    class NeverEnvironment:
        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]], dict[str, object]]:
            del seed
            raise AssertionError("environment reset must not run")

        def step(self, action: object) -> FrozenStep:
            del action
            raise AssertionError("environment step must not run")

        def close(self) -> None:
            return None

    def environment_factory() -> NeverEnvironment:
        nonlocal created
        created += 1
        raise AssertionError("environment must not be constructed")

    def runtime_report() -> NativeRuntimeReport:
        raise AssertionError("runtime/environment boundary must not run")

    monkeypatch.setattr(vertical_smoke, "create_nonproduction_native_fixture", fixture)
    monkeypatch.setattr(vertical_smoke, "import_repo_store", imported)
    monkeypatch.setattr(vertical_smoke, "load_paper_view", loaded)
    monkeypatch.setattr(vertical_smoke, "_run_worker", cancelled)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected cancellation"):
            run_model_smoke(
                "dp_cnn",
                output,
                fixture=True,
                dependencies=SmokeDependencies(
                    paper_python=fake_python,
                    runtime_report=runtime_report,
                    environment_factory=environment_factory,
                ),
            )
        assert created == 0
        assert not output.exists()
        assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    finally:
        fake_python.unlink(missing_ok=True)
        shutil.rmtree(output, ignore_errors=True)


def test_vertical_smoke_preserves_stale_final_and_staging_markers(
    canonical_test_root: Path,
) -> None:
    namespace = canonical_test_root.name
    output = runtime_artifact_root() / f"{namespace}-stale-output"
    output.mkdir()
    final_marker = output / "foreign"
    final_marker.write_bytes(b"final-foreign")
    fake_python = runtime_artifact_root() / f"{namespace}-fake-python-stale"
    fake_python.write_text("fixture", encoding="utf-8")
    dependencies = SmokeDependencies(paper_python=fake_python)
    staging: Path | None = None
    try:
        with pytest.raises(ModelSmokeError, match="already exists"):
            run_model_smoke("dp_cnn", output, fixture=True, dependencies=dependencies)
        assert final_marker.read_bytes() == b"final-foreign"

        shutil.rmtree(output)
        token = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
        staging = output.with_name(f".{output.name}.tmp-{token}")
        staging.mkdir()
        staging_marker = staging / "foreign"
        staging_marker.write_bytes(b"staging-foreign")
        with pytest.raises(ModelSmokeError, match="staging already exists"):
            run_model_smoke("dp_cnn", output, fixture=True, dependencies=dependencies)
        assert not output.exists()
        assert staging_marker.read_bytes() == b"staging-foreign"
    finally:
        fake_python.unlink(missing_ok=True)
        shutil.rmtree(output, ignore_errors=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def test_typed_result_requires_nonproduction_finite_float32_rollout(tmp_path: Path) -> None:
    _, reload_receipt, _, _ = _receipt(tmp_path)
    identity = reload_receipt["identity"]
    assert isinstance(identity, dict)
    result: dict[str, object] = {
        "schema": MODEL_SMOKE_SCHEMA,
        "artifact_type": "bounded_fixture_model_vertical_slice",
        "model": "dp_cnn",
        "fixture": True,
        "production_eligible": False,
        "comparison_eligible": False,
        "result_status": "ineligible_fixture",
        "identity": identity,
        "native_fixture": {
            "format": "LeRobot-0.4.4-v3.0",
            "fps": 10,
            "episodes": 1,
            "frames": 16,
            "canonical_digest": "a" * 64,
            "root_digest": "f" * 64,
            "explicit_nonproduction_marker": "fixture-source/synthetic-fixture.NON_PRODUCTION.json",
            "import_validated": True,
        },
        "training": {
            "optimizer_updates": 1,
            "loss": 1.25,
            "policy_class": "DiffusionUnetHybridImagePolicy",
        },
        "checkpoint": {
            "path": "training/checkpoint.ckpt",
            "sha256": "1" * 64,
            "config_path": "training/resolved_config.json",
            "config_sha256": "2" * 64,
            "strict_identity_reload": True,
        },
        "rollout": {
            "seed": 100000,
            "steps": 1,
            "action": [0.25, -0.25],
            "action_dtype": "float32",
            "action_shape": [2],
            "action_finite": True,
            "dxy": 0.1,
            "dyaw": 0.2,
            "terminated": False,
            "truncated": True,
            "frozen_environment_manifest_sha256": "d" * 64,
        },
        "runtime_lock_sha256": "c" * 64,
        "teardown": {
            "environment_closed": True,
            "worker_processes_reaped": True,
            "temporary_observation_files_removed": True,
            "transaction_staging_published": True,
        },
    }
    assert validate_model_smoke_result(result, expected_model="dp_cnn") == result
    for mutation in (
        {**result, "production_eligible": True},
        {
            **result,
            "rollout": {
                **cast("dict[str, object]", result["rollout"]),
                "action": [0.0, float("nan")],
            },
        },
        {
            **result,
            "teardown": {
                **cast("dict[str, object]", result["teardown"]),
                "environment_closed": False,
            },
        },
    ):
        with pytest.raises(ModelSmokeError, match=r"eligibility|rollout|teardown"):
            validate_model_smoke_result(mutation, expected_model="dp_cnn")


def test_reload_receipt_rejects_unknown_fields_and_path_escape(tmp_path: Path) -> None:
    path, receipt, index, store = _receipt(tmp_path)
    path.write_text(json.dumps({**receipt, "production": True}), encoding="utf-8")
    with pytest.raises(SmokeIdentityError, match="fields"):
        _validate(path, index, store)

    path.write_text(json.dumps({**receipt, "checkpoint": "../checkpoint.ckpt"}), encoding="utf-8")
    with pytest.raises(SmokeIdentityError, match="checkpoint"):
        _validate(path, index, store)
