"""Transactional native-fixture vertical smoke orchestration across locked runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..data.importer import import_repo_store
from ..data.paper_view import LoadedPaperView
from ..data.paper_view_reader import load_paper_view
from ..evaluation.frozen_env import FrozenStep, load_frozen_pusht, validate_action
from ..native_runtime import NativeRuntimeReport, native_runtime_report
from ..workspace import PROJECT_ROOT, runtime_artifact_root
from .artifacts import ArtifactIndex
from .smoke_contract import MODEL_SMOKE_SCHEMA

_MODEL_SPECS = {
    "dp_cnn": (
        "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy",
        "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace",
        2,
        16,
        8,
    ),
    "dp_transformer": (
        "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy",
        "diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace",
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
    "lstm_gmm": (
        "diffusion_policy.policy.robomimic_image_policy.RobomimicImagePolicy",
        "diffusion_policy.workspace.train_robomimic_image_workspace.TrainRobomimicImageWorkspace",
        10,
        10,
        1,
    ),
}
_IDENTITY_FIELDS = {
    "model",
    "policy_target",
    "workspace_target",
    "observation_steps",
    "horizon",
    "executed_actions",
    "optimizer_updates",
    "dataset_digest",
    "split_digest",
    "runtime_lock_digest",
    "environment_manifest_digest",
    "stanford_commit",
    "robomimic_commit",
}
_RECEIPT_FIELDS = {
    "schema",
    "phase",
    "model",
    "fixture",
    "production_eligible",
    "comparison_eligible",
    "identity",
    "store_identity",
    "checkpoint",
    "checkpoint_sha256",
    "config",
    "config_sha256",
    "reload_verified",
    "policy_class",
    "optimizer_updates",
    "loss",
}
_PAPER_PYTHON = (
    PROJECT_ROOT / "04_experiments/so101_pusht_benchmark/cache/envs/paper-baselines/bin/python"
)


class SmokeIdentityError(ValueError):
    """Raised when a fixture checkpoint, config, or model identity is not exact."""


class ModelSmokeError(RuntimeError):
    """Raised when a vertical smoke cannot publish one complete result."""


class _Environment(Protocol):
    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]: ...

    def step(self, action: object) -> FrozenStep: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SmokeDependencies:
    """Owned seams used to prove ordering and teardown without replacing model behavior."""

    paper_python: Path = _PAPER_PYTHON
    runtime_report: Callable[[], NativeRuntimeReport] = native_runtime_report
    environment_factory: Callable[[], _Environment] = lambda: load_frozen_pusht(max_steps=1)


@dataclass(frozen=True, slots=True)
class ReloadValidationContext:
    artifact_root: Path
    index: ArtifactIndex
    artifact_id: str
    store_path: Path
    store_view: LoadedPaperView


@dataclass(frozen=True, slots=True)
class _WorkerRequest:
    model: str
    expected_model: str
    store: Path
    root: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SmokeIdentityError(f"{label} identity is not a lowercase SHA-256 digest")
    return value


def _safe_member(root: Path, value: object, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise SmokeIdentityError(f"{label} path is invalid")
    path = root / value
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise SmokeIdentityError(f"{label} artifact is missing or unsafe")
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise SmokeIdentityError(f"{label} path escapes the artifact root")
    return resolved


def expected_smoke_identity(model: str, identity: object) -> dict[str, object]:
    """Validate the complete compact identity emitted by the paper-runtime worker."""
    spec = _MODEL_SPECS.get(model)
    if spec is None:
        raise SmokeIdentityError(f"unknown model identity: {model}")
    if not isinstance(identity, dict):
        raise SmokeIdentityError("model identity fields are not exact")
    raw = cast("dict[str, object]", identity)
    if set(raw) != _IDENTITY_FIELDS:
        raise SmokeIdentityError("model identity fields are not exact")
    policy, workspace, observation_steps, horizon, executed_actions = spec
    expected = {
        "model": model,
        "policy_target": policy,
        "workspace_target": workspace,
        "observation_steps": observation_steps,
        "horizon": horizon,
        "executed_actions": executed_actions,
        "optimizer_updates": 1,
        "stanford_commit": "5ba07ac6661db573af695b419a7947ecb704690f",
        "robomimic_commit": "62ed2de905caeb9133136e4d14d810a8b6baa96c",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise SmokeIdentityError(f"model identity mismatch: {key}")
    for key in (
        "dataset_digest",
        "split_digest",
        "runtime_lock_digest",
        "environment_manifest_digest",
    ):
        _digest(raw[key], key)
    return raw


def validate_reload_receipt(
    path: Path, *, expected_model: str, context: ReloadValidationContext
) -> dict[str, object]:
    """Validate model/config/checkpoint identity before environment construction."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeIdentityError("reload receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SmokeIdentityError("reload receipt must be an object")
    receipt = cast("dict[str, object]", value)
    if set(receipt) != _RECEIPT_FIELDS:
        raise SmokeIdentityError("reload receipt fields are not exact")
    if receipt.get("schema") != MODEL_SMOKE_SCHEMA or receipt.get("phase") != "strict_reload":
        raise SmokeIdentityError("reload receipt schema is invalid")
    if receipt.get("model") != expected_model:
        raise SmokeIdentityError("model identity mismatch before environment creation")
    if (
        receipt.get("fixture") is not True
        or receipt.get("production_eligible") is not False
        or receipt.get("comparison_eligible") is not False
        or receipt.get("reload_verified") is not True
        or receipt.get("optimizer_updates") != 1
    ):
        raise SmokeIdentityError("fixture/reload identity is invalid")
    identity = expected_smoke_identity(expected_model, receipt.get("identity"))
    record = context.index.record(context.artifact_id)
    record_identity_raw = record.get("identity")
    if not isinstance(record_identity_raw, dict):
        raise SmokeIdentityError("artifact-index identity is missing")
    record_identity = cast("dict[str, object]", record_identity_raw)
    if any(record_identity.get(key) != value for key, value in identity.items()):
        raise SmokeIdentityError("dataset/model identity differs from artifact-index identity")
    if (
        record.get("deployment_scope") != "simulation_only"
        or record.get("training_eligible") is not False
        or record.get("comparison_eligible") is not False
        or record.get("result_status") != "ineligible_fixture"
    ):
        raise SmokeIdentityError("artifact-index fixture lifecycle identity mismatch")

    checkpoint = _safe_member(context.artifact_root, receipt.get("checkpoint"), "checkpoint")
    config = _safe_member(context.artifact_root, receipt.get("config"), "config")
    if context.index.verify(context.artifact_id, "checkpoint") != checkpoint:
        raise SmokeIdentityError("checkpoint path differs from artifact-index identity")
    if context.index.verify(context.artifact_id, "config") != config:
        raise SmokeIdentityError("config path differs from artifact-index identity")
    if _sha256(checkpoint) != _digest(receipt.get("checkpoint_sha256"), "checkpoint"):
        raise SmokeIdentityError("checkpoint digest identity mismatch")
    if _sha256(config) != _digest(receipt.get("config_sha256"), "config"):
        raise SmokeIdentityError("config digest identity mismatch")

    canonical = _digest(context.store_view.manifest.get("canonical_digest"), "store canonical")
    root_digest = _digest(context.store_view.manifest.get("root_digest"), "store root")
    expected_split = hashlib.sha256(f"ineligible-fixture:{canonical}".encode()).hexdigest()
    store_identity_raw = receipt.get("store_identity")
    if not isinstance(store_identity_raw, dict):
        raise SmokeIdentityError("store identity is missing")
    store_identity = cast("dict[str, object]", store_identity_raw)
    if set(store_identity) != {
        "canonical_digest",
        "root_digest",
        "split_digest",
        "manifest_sha256",
        "splits_sha256",
    }:
        raise SmokeIdentityError("store identity fields are not exact")
    manifest_path = _safe_member(
        context.store_path.parent,
        context.store_path.name + "/manifest.json",
        "store manifest",
    )
    splits_path = _safe_member(
        context.store_path.parent,
        context.store_path.name + "/splits.json",
        "store splits",
    )
    if (
        identity["dataset_digest"] != canonical
        or identity["split_digest"] != expected_split
        or store_identity.get("canonical_digest") != canonical
        or store_identity.get("root_digest") != root_digest
        or store_identity.get("split_digest") != expected_split
        or store_identity.get("manifest_sha256") != _sha256(manifest_path)
        or store_identity.get("splits_sha256") != _sha256(splits_path)
    ):
        raise SmokeIdentityError("dataset/canonical/split/root store identity mismatch")
    try:
        config_value: object = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeIdentityError("config identity is not valid JSON") from exc
    if not isinstance(config_value, dict):
        raise SmokeIdentityError("config identity must be an object")
    config_raw = cast("dict[str, object]", config_value)
    policy_raw = config_raw.get("policy")
    task_raw = config_raw.get("task")
    dataset_raw: object = None
    if isinstance(task_raw, dict):
        dataset_raw = cast("dict[str, object]", task_raw).get("dataset")
    dataset_path: object = None
    if isinstance(dataset_raw, dict):
        dataset_path = cast("dict[str, object]", dataset_raw).get("zarr_path")
    if (
        config_raw.get("name") != expected_model
        or config_raw.get("_target_") != identity["workspace_target"]
        or not isinstance(policy_raw, dict)
        or cast("dict[str, object]", policy_raw).get("_target_") != identity["policy_target"]
        or not isinstance(dataset_path, str)
        or Path(dataset_path).resolve() != context.store_path.resolve()
    ):
        raise SmokeIdentityError("config model/dataset identity mismatch")
    loss = receipt.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not np.isfinite(loss):
        raise SmokeIdentityError("optimizer loss is not finite")
    expected_class = str(identity["policy_target"]).rsplit(".", 1)[1]
    if receipt.get("policy_class") != expected_class:
        raise SmokeIdentityError("model policy class identity mismatch")
    return receipt


def load_inference_action(root: Path, *, expected_model: str) -> NDArray[np.float32]:
    """Load and validate exact inference metadata and float32 bytes before any environment exists."""
    path = root / "inference.json"
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSmokeError("inference action metadata is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ModelSmokeError("inference action metadata must be an object")
    metadata = cast("dict[str, object]", value)
    if (
        set(metadata)
        != {
            "schema",
            "model",
            "checkpoint_reloaded",
            "action_dtype",
            "action_shape",
            "action",
        }
        or metadata.get("schema") != MODEL_SMOKE_SCHEMA
        or metadata.get("model") != expected_model
        or metadata.get("checkpoint_reloaded") is not True
        or metadata.get("action_dtype") != "float32"
        or metadata.get("action_shape") != [2]
    ):
        raise ModelSmokeError("inference action identity/shape is invalid")
    action_path = root / "action.bin"
    if action_path.is_symlink() or not action_path.is_file() or action_path.stat().st_size != 8:
        raise ModelSmokeError("inference action bytes must be exact float32[2]")
    action = np.fromfile(action_path, dtype=np.float32)
    checked = validate_action(action)
    declared = metadata.get("action")
    if not isinstance(declared, list) or declared != checked.tolist():
        raise ModelSmokeError("inference action metadata differs from action bytes")
    return checked


def validate_model_smoke_result(value: object, *, expected_model: str) -> dict[str, object]:
    """Validate the persisted result's exact typed fixture and rollout contract."""
    if not isinstance(value, dict):
        raise ModelSmokeError("model smoke result must be an object")
    result = cast("dict[str, object]", value)
    expected_root = {
        "schema",
        "artifact_type",
        "model",
        "fixture",
        "production_eligible",
        "comparison_eligible",
        "result_status",
        "identity",
        "native_fixture",
        "training",
        "checkpoint",
        "rollout",
        "runtime_lock_sha256",
        "teardown",
    }
    if set(result) != expected_root:
        raise ModelSmokeError("model smoke result fields are not exact")
    if (
        result.get("schema") != MODEL_SMOKE_SCHEMA
        or result.get("artifact_type") != "bounded_fixture_model_vertical_slice"
        or result.get("model") != expected_model
        or result.get("fixture") is not True
        or result.get("production_eligible") is not False
        or result.get("comparison_eligible") is not False
        or result.get("result_status") != "ineligible_fixture"
    ):
        raise ModelSmokeError("model smoke result eligibility/schema is invalid")
    identity = expected_smoke_identity(expected_model, result.get("identity"))
    fixture = result.get("native_fixture")
    training = result.get("training")
    checkpoint = result.get("checkpoint")
    rollout = result.get("rollout")
    teardown = result.get("teardown")
    if not all(
        isinstance(item, dict) for item in (fixture, training, checkpoint, rollout, teardown)
    ):
        raise ModelSmokeError("model smoke result sections must be objects")
    fixture_values = cast("dict[str, object]", fixture)
    training_values = cast("dict[str, object]", training)
    checkpoint_values = cast("dict[str, object]", checkpoint)
    rollout_values = cast("dict[str, object]", rollout)
    teardown_values = cast("dict[str, object]", teardown)
    if (
        set(fixture_values)
        != {
            "format",
            "fps",
            "episodes",
            "frames",
            "canonical_digest",
            "root_digest",
            "explicit_nonproduction_marker",
            "import_validated",
        }
        or fixture_values.get("format") != "LeRobot-0.4.4-v3.0"
        or fixture_values.get("fps") != 10
        or fixture_values.get("episodes") != 1
        or fixture_values.get("frames") != 16
        or fixture_values.get("import_validated") is not True
        or not str(fixture_values.get("explicit_nonproduction_marker", "")).endswith(
            "NON_PRODUCTION.json"
        )
    ):
        raise ModelSmokeError("model smoke native fixture section is invalid")
    _digest(fixture_values.get("canonical_digest"), "fixture canonical")
    _digest(fixture_values.get("root_digest"), "fixture root")
    loss = training_values.get("loss")
    if (
        set(training_values) != {"optimizer_updates", "loss", "policy_class"}
        or training_values.get("optimizer_updates") != 1
        or isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not np.isfinite(loss)
        or training_values.get("policy_class") != str(identity["policy_target"]).rsplit(".", 1)[1]
    ):
        raise ModelSmokeError("model smoke training section is invalid")
    if (
        set(checkpoint_values)
        != {"path", "sha256", "config_path", "config_sha256", "strict_identity_reload"}
        or checkpoint_values.get("strict_identity_reload") is not True
        or not isinstance(checkpoint_values.get("path"), str)
        or not isinstance(checkpoint_values.get("config_path"), str)
    ):
        raise ModelSmokeError("model smoke checkpoint section is invalid")
    _digest(checkpoint_values.get("sha256"), "result checkpoint")
    _digest(checkpoint_values.get("config_sha256"), "result config")
    action = rollout_values.get("action")
    if not isinstance(action, list):
        raise ModelSmokeError("model smoke rollout section is invalid")
    action_values = cast("list[object]", action)
    if (
        set(rollout_values)
        != {
            "seed",
            "steps",
            "action",
            "action_dtype",
            "action_shape",
            "action_finite",
            "dxy",
            "dyaw",
            "terminated",
            "truncated",
            "frozen_environment_manifest_sha256",
        }
        or rollout_values.get("seed") != 100000
        or rollout_values.get("steps") != 1
        or rollout_values.get("action_dtype") != "float32"
        or rollout_values.get("action_shape") != [2]
        or rollout_values.get("action_finite") is not True
        or len(action_values) != 2
        or not all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and np.isfinite(item)
            and -1.0 <= item <= 1.0
            for item in action_values
        )
        or type(rollout_values.get("terminated")) is not bool
        or type(rollout_values.get("truncated")) is not bool
    ):
        raise ModelSmokeError("model smoke rollout section is invalid")
    for metric in ("dxy", "dyaw"):
        item = rollout_values.get(metric)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item):
            raise ModelSmokeError(f"model smoke rollout {metric} is invalid")
    if (
        rollout_values.get("frozen_environment_manifest_sha256")
        != identity["environment_manifest_digest"]
    ):
        raise ModelSmokeError("model smoke frozen environment identity mismatch")
    if result.get("runtime_lock_sha256") != identity["runtime_lock_digest"]:
        raise ModelSmokeError("model smoke runtime identity mismatch")
    if teardown_values != {
        "environment_closed": True,
        "worker_processes_reaped": True,
        "temporary_observation_files_removed": True,
        "transaction_staging_published": True,
    }:
        raise ModelSmokeError("model smoke teardown receipt is invalid")
    return result


def load_model_smoke_result(path: Path, *, expected_model: str) -> dict[str, object]:
    """Load and type-check one published model-smoke result artifact."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSmokeError("model smoke result is not valid JSON") from exc
    return validate_model_smoke_result(value, expected_model=expected_model)


def create_nonproduction_native_fixture(root: Path, *, frames: int = 16) -> Path:
    """Create a real LeRobot 0.4.4 fixture marked permanently non-production."""
    if type(frames) is not int or frames < 16:
        raise ModelSmokeError("model smoke fixture requires at least 16 frames")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features: dict[str, object] = {
        "observation.images.cam_top": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_side": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (5,)},
        "action": {"dtype": "float32", "shape": (2,)},
    }
    dataset = LeRobotDataset.create(
        "local/todo10-explicitly-nonproduction-fixture",
        fps=10,
        features=features,
        root=root,
        vcodec="h264",
    )
    top = np.full((224, 224, 3), 32, dtype=np.uint8)
    side = np.full((224, 224, 3), 192, dtype=np.uint8)
    for frame in range(frames):
        dataset.add_frame(
            {
                "observation.images.cam_top": top,
                "observation.images.cam_side": side,
                "observation.state": np.asarray(
                    [frame / 100, -frame / 120, frame / 140, -frame / 160, frame / 180],
                    dtype=np.float32,
                ),
                "action": np.asarray(
                    [0.35 * np.sin(frame / 4), 0.35 * np.cos(frame / 4)], dtype=np.float32
                ),
                "task": "pushT fixture - NON-PRODUCTION",
            }
        )
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    marker = {
        "schema": 1,
        "artifact_type": "synthetic_fixture",
        "production_eligible": False,
        "comparison_eligible": False,
        "reason": "synthetic_fixture_not_comparison_eligible",
    }
    (root / "synthetic-fixture.NON_PRODUCTION.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _run_worker(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    (cwd / f"worker-{command[3]}.stdout.txt").write_text(stdout, encoding="utf-8")
    (cwd / f"worker-{command[3]}.stderr.txt").write_text(stderr, encoding="utf-8")
    if process.returncode != 0:
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no worker diagnostic"
        raise ModelSmokeError(f"paper-runtime {command[3]} failed: {detail}")


def _paper_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(PROJECT_ROOT / "03_code/so101_pusht_benchmark/src"),
        PYTHONDONTWRITEBYTECODE="1",
        WANDB_MODE="offline",
    )
    return environment


def _worker_command(python: Path, phase: str, request: _WorkerRequest) -> list[str]:
    return [
        str(python),
        "-m",
        "so101_pusht_benchmark.training.smoke_worker",
        phase,
        "--model",
        request.model,
        "--expected-model",
        request.expected_model,
        "--store",
        str(request.store),
        "--root",
        str(request.root),
    ]


def _validate_output(output: Path) -> tuple[Path, Path]:
    root = runtime_artifact_root().resolve()
    absolute = output.resolve(strict=False)
    if (
        output.is_symlink()
        or absolute == root
        or root not in absolute.resolve(strict=False).parents
    ):
        raise ModelSmokeError("model smoke output must be beneath the canonical artifact root")
    if not absolute.parent.is_dir() or absolute.parent.is_symlink():
        raise ModelSmokeError("model smoke output parent must be an existing real directory")
    if absolute.exists() or absolute.is_symlink():
        raise ModelSmokeError(f"model smoke output already exists: {absolute}")
    token = hashlib.sha256(str(absolute).encode()).hexdigest()[:12]
    staging = absolute.with_name(f".{absolute.name}.tmp-{token}")
    if staging.exists() or staging.is_symlink():
        raise ModelSmokeError(f"model smoke staging already exists: {staging}")
    return absolute, staging


def _fixture_observation(view: LoadedPaperView) -> dict[str, NDArray[np.generic]]:
    result: dict[str, NDArray[np.generic]] = {}
    for key in ("cam_top", "cam_side", "agent_pos"):
        value = view.arrays.get(key)
        if not isinstance(value, np.ndarray) or value.shape[0] < 1:
            raise ModelSmokeError(f"fixture observation {key} is unavailable")
        result[key] = cast("NDArray[np.generic]", value[0])
    return result


def _write_observation(root: Path, observation: dict[str, NDArray[np.generic]]) -> None:
    for key in ("cam_top", "cam_side", "agent_pos"):
        np.save(root / f"rollout-{key}.npy", observation[key], allow_pickle=False)


def _cleanup_observation(root: Path) -> None:
    for key in ("cam_top", "cam_side", "agent_pos"):
        (root / f"rollout-{key}.npy").unlink(missing_ok=True)


def run_model_smoke(
    model: str,
    output: Path,
    *,
    fixture: bool,
    reload_model: str | None = None,
    dependencies: SmokeDependencies | None = None,
) -> Path:
    """Run import, update, strict reload, frozen reset/inference/step, and atomic publish."""
    if model not in _MODEL_SPECS:
        raise ModelSmokeError(f"unknown model: {model}")
    if not fixture:
        raise ModelSmokeError("model-smoke currently requires --fixture")
    expected_model = model if reload_model is None else reload_model
    if expected_model not in _MODEL_SPECS:
        raise ModelSmokeError(f"unknown reload model: {expected_model}")
    selected = SmokeDependencies() if dependencies is None else dependencies
    if not selected.paper_python.is_file():
        raise ModelSmokeError("locked paper-baselines Python is unavailable")
    final, staging = _validate_output(output)
    staging.mkdir()
    environment: _Environment | None = None
    closed = False
    try:
        source = create_nonproduction_native_fixture(staging / "fixture-source")
        store = staging / "fixture-store"
        captured_stdout, captured_stderr = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
        ):
            imported = import_repo_store(source, store)
        (staging / "fixture-import.stdout.txt").write_text(
            captured_stdout.getvalue(), encoding="utf-8"
        )
        (staging / "fixture-import.stderr.txt").write_text(
            captured_stderr.getvalue(), encoding="utf-8"
        )
        if imported != 0:
            raise ModelSmokeError("native fixture import/validation failed")
        view = load_paper_view(store)
        if (
            view.manifest.get("training_eligible") is not False
            or view.splits.get("training_eligible") is not False
            or view.splits.get("reason") != "synthetic_fixture_not_comparison_eligible"
        ):
            raise ModelSmokeError("fixture import is not explicitly non-production")

        _write_observation(staging, _fixture_observation(view))
        worker_request = _WorkerRequest(model, expected_model, store, staging)
        train_command = _worker_command(selected.paper_python, "train", worker_request)
        _run_worker(train_command, cwd=staging, environment=_paper_environment())
        reload_path = staging / "reload.json"
        index = ArtifactIndex(staging / "artifact-index.json", staging)
        reload_receipt = validate_reload_receipt(
            reload_path,
            expected_model=expected_model,
            context=ReloadValidationContext(
                staging,
                index,
                "fixture-model-smoke",
                store,
                view,
            ),
        )
        if expected_model != model:
            raise SmokeIdentityError("model identity mismatch before environment creation")
        checked = load_inference_action(staging, expected_model=model)
        _cleanup_observation(staging)

        runtime = selected.runtime_report()
        environment = selected.environment_factory()
        environment.reset(seed=100000)
        step = environment.step(checked)
        environment.close()
        closed = True
        environment = None
        _cleanup_observation(staging)

        identity = cast("dict[str, object]", reload_receipt["identity"])
        result = {
            "schema": MODEL_SMOKE_SCHEMA,
            "artifact_type": "bounded_fixture_model_vertical_slice",
            "model": model,
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
                "canonical_digest": view.manifest["canonical_digest"],
                "root_digest": view.manifest["root_digest"],
                "explicit_nonproduction_marker": "fixture-source/synthetic-fixture.NON_PRODUCTION.json",
                "import_validated": True,
            },
            "training": {
                "optimizer_updates": 1,
                "loss": reload_receipt["loss"],
                "policy_class": reload_receipt["policy_class"],
            },
            "checkpoint": {
                "path": reload_receipt["checkpoint"],
                "sha256": reload_receipt["checkpoint_sha256"],
                "config_path": reload_receipt["config"],
                "config_sha256": reload_receipt["config_sha256"],
                "strict_identity_reload": True,
            },
            "rollout": {
                "seed": 100000,
                "steps": 1,
                "action": checked.tolist(),
                "action_dtype": "float32",
                "action_shape": [2],
                "action_finite": True,
                "dxy": step.info["dxy"],
                "dyaw": step.info["dyaw"],
                "terminated": step.terminated,
                "truncated": step.truncated,
                "frozen_environment_manifest_sha256": identity["environment_manifest_digest"],
            },
            "runtime_lock_sha256": runtime["lock_sha256"],
            "teardown": {
                "environment_closed": closed,
                "worker_processes_reaped": True,
                "temporary_observation_files_removed": True,
                "transaction_staging_published": True,
            },
        }
        validate_model_smoke_result(result, expected_model=model)
        (staging / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(final)
        return final / "result.json"
    except BaseException:
        if environment is not None:
            environment.close()
        _cleanup_observation(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise
