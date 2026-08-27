"""Active native pushT-so100 command-line contract and preflight surfaces."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

import yaml

from .collection.native import (
    DEVICE_PROBE_ENV,
    DISPLAY_PROBE_ENV,
    NativeCollectionError,
    NativeCollectionPlan,
    NativeCollectionRequest,
    launch_native_collection,
    preflight_native_collection,
)
from .native_runtime import (
    NativeRuntimeError,
    NativeRuntimeReport,
    native_runtime_report,
)
from .training.artifacts import ArtifactIndex
from .workspace import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    WorkspacePolicyError,
    load_workspace_policy,
)


RuntimeReport = NativeRuntimeReport
_BENCHMARK_CONFIG = PACKAGE_ROOT / "configs/benchmark/pusht_so100_native_v1.yaml"
_COLLECTION_CONFIG = PACKAGE_ROOT / "configs/collection/pusht_so100_f710_native_v1.yaml"
_EXPORT_CONFIG = PACKAGE_ROOT / "configs/export/pusht_so100_native_v1.yaml"
_DEFAULT_DATASET_ROOT = (
    PACKAGE_ROOT.parents[1] / "04_experiments/so101_pusht_benchmark/datasets/pusht_so100_native"
)


class NativeCliError(ValueError):
    """Raised when a requested native route is historical, malformed, or inconsistent."""


class NativeConfig(TypedDict):
    schema: int
    status: str
    contract_schema: str
    runtime_lock: str


class ImageConfig(TypedDict):
    dtype: str
    shape: list[int]


class AgentPositionConfig(ImageConfig):
    order: list[str]


class ActionConfig(ImageConfig):
    meaning: str
    bounds: list[float]


class BenchmarkObservationConfig(TypedDict):
    cam_top: ImageConfig
    cam_side: ImageConfig
    agent_pos: AgentPositionConfig


class BenchmarkConfig(NativeConfig):
    identifier: str
    deployment_scope: str
    upstream_environment: str
    fps: int
    horizon: int
    observation: BenchmarkObservationConfig
    action: ActionConfig
    policy_allowlist: list[str]


class ControllerAxes(TypedDict):
    x: int
    y: int
    rotation: int


class ControllerButtons(TypedDict):
    z_up: int
    z_down: int
    reset: int
    record_toggle: int
    exit: int


class ControllerConfig(TypedDict):
    axes: ControllerAxes
    buttons: ControllerButtons
    deadzone: float
    move_speed: float
    rotation_speed: float
    button_debounce_seconds: float


class DatasetFieldConfig(ImageConfig):
    key: str


class DatasetImagesConfig(TypedDict):
    cam_top: ImageConfig
    cam_side: ImageConfig


class CollectionDatasetConfig(TypedDict):
    format: str
    version: str
    images: DatasetImagesConfig
    state: DatasetFieldConfig
    action: DatasetFieldConfig


class CollectionConfig(NativeConfig):
    source: str
    fps: int
    controller: ControllerConfig
    dataset: CollectionDatasetConfig


class ExportConfig(NativeConfig):
    format: str
    fps: int
    keys: list[str]
    transforms: str


ActiveConfig = BenchmarkConfig | CollectionConfig | ExportConfig
ConfigRole = Literal["benchmark", "collection", "export"]

_BENCHMARK_EXPECTED: BenchmarkConfig = {
    "schema": 1,
    "status": "active",
    "contract_schema": "pusht-so100-native-v1",
    "identifier": "pusht_so100_native_v1",
    "deployment_scope": "simulation_only",
    "upstream_environment": "05_references/external_repos/pushT-so100",
    "runtime_lock": "environments/sim-runtime.lock",
    "fps": 10,
    "horizon": 300,
    "observation": {
        "cam_top": {"dtype": "uint8", "shape": [3, 224, 224]},
        "cam_side": {"dtype": "uint8", "shape": [3, 224, 224]},
        "agent_pos": {
            "dtype": "float32",
            "shape": [5],
            "order": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": [2],
        "meaning": "absolute_mocap_xy",
        "bounds": [-1.0, 1.0],
    },
    "policy_allowlist": ["cam_top", "cam_side", "agent_pos", "action"],
}
_COLLECTION_EXPECTED: CollectionConfig = {
    "schema": 1,
    "status": "active",
    "contract_schema": "pusht-so100-native-v1",
    "source": "frozen_pusht_so100_f710",
    "runtime_lock": "environments/sim-runtime.lock",
    "fps": 10,
    "controller": {
        "axes": {"x": 0, "y": 1, "rotation": 3},
        "buttons": {"z_up": 4, "z_down": 0, "reset": 3, "record_toggle": 1, "exit": 7},
        "deadzone": 0.1,
        "move_speed": 0.05,
        "rotation_speed": 1.0,
        "button_debounce_seconds": 0.3,
    },
    "dataset": {
        "format": "lerobot",
        "version": "0.4.4",
        "images": {
            "cam_top": {"dtype": "uint8", "shape": [224, 224, 3]},
            "cam_side": {"dtype": "uint8", "shape": [224, 224, 3]},
        },
        "state": {"key": "agent_pos", "dtype": "float32", "shape": [5]},
        "action": {"key": "action", "dtype": "float32", "shape": [2]},
    },
}
_EXPORT_EXPECTED: ExportConfig = {
    "schema": 1,
    "status": "active",
    "contract_schema": "pusht-so100-native-v1",
    "runtime_lock": "environments/sim-runtime.lock",
    "format": "lerobot-native",
    "fps": 10,
    "keys": [
        "cam_top",
        "cam_side",
        "agent_pos",
        "action",
        "timestamp",
        "episode_id",
        "frame_index",
        "episode_ends",
    ],
    "transforms": "forbidden",
}


def command_parser() -> argparse.ArgumentParser:
    """Return only the active clean-restart command surface."""
    parser = argparse.ArgumentParser(prog="so101-pusht-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-contract")
    validate.add_argument("--config", type=Path, default=_BENCHMARK_CONFIG)

    inspect = commands.add_parser("inspect-env")
    inspect.add_argument("--native-pusht-so100", action="store_true", required=True)

    collect = commands.add_parser("collect-native")
    collection_mode = collect.add_mutually_exclusive_group(required=True)
    collection_mode.add_argument("--preflight", action="store_true")
    collection_mode.add_argument("--launch", action="store_true")
    collect.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    collect.add_argument("--fps", type=int, default=10)
    collect.add_argument("--move-speed", type=float, default=0.05)
    collect.add_argument("--rotation-speed", type=float, default=1.0)
    collect.add_argument("--config", type=Path, default=_BENCHMARK_CONFIG)
    collect.add_argument("--collection-config", type=Path, default=_COLLECTION_CONFIG)

    import_native = commands.add_parser("import-native")
    import_native.add_argument("--repo", type=Path, required=True)
    import_native.add_argument("--output", type=Path, required=True)
    import_native.add_argument("--config", type=Path, default=_EXPORT_CONFIG)

    export = commands.add_parser("export-native")
    export.add_argument("--preflight", action="store_true")
    export.add_argument("--root", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument("--config", type=Path, default=_EXPORT_CONFIG)

    freeze = commands.add_parser(
        "freeze-experiment",
        description=(
            "Plan or freeze the user-selected episode budget. Metadata dry-run planning "
            "does not probe the F710, create artifacts, or start training."
        ),
    )
    freeze_input = freeze.add_mutually_exclusive_group(required=True)
    freeze_input.add_argument("--source", type=Path)
    freeze_input.add_argument(
        "--metadata",
        type=Path,
        help="native LeRobot meta/info.json used only with --dry-run",
    )
    freeze.add_argument("--output", type=Path)
    freeze.add_argument("--experiment-config", type=Path, required=True)
    freeze.add_argument("--sessions", type=Path)
    freeze.add_argument(
        "--dry-run",
        action="store_true",
        help="plan an immutable split from collection metadata without writing output",
    )

    train = commands.add_parser(
        "train-model",
        description=(
            "Choose a non-final one-update smoke proof or full production training. "
            "Full production requires the frozen manifest and an approved paper-profile "
            "optimizer-update bound."
        ),
    )
    train.add_argument(
        "--model", choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"), required=True
    )
    train.add_argument("--paper-view", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    train.add_argument("--artifact-id", required=True)
    train.add_argument("--artifact-index", type=Path, required=True)
    train.add_argument("--paper-profiles", type=Path)
    train.add_argument(
        "--preflight",
        action="store_true",
        help="validate full-production store/config/identity without creating output or training",
    )
    training_mode = train.add_mutually_exclusive_group(required=True)
    training_mode.add_argument(
        "--smoke",
        action="store_true",
        help="compatibility alias for --smoke-mode fixture (one update, never comparison eligible)",
    )
    training_mode.add_argument(
        "--smoke-mode",
        choices=("fixture", "production"),
        help="one-update proof only; production smoke is never export/evaluation eligible",
    )
    training_mode.add_argument(
        "--full-production",
        action="store_true",
        help="run the locked full workspace to the explicit --max-updates bound",
    )
    train.add_argument(
        "--max-updates",
        type=int,
        help="required with --full-production and must equal the selected paper profile",
    )

    model_smoke = commands.add_parser(
        "model-smoke",
        description=(
            "Run a transactional non-production native fixture import, one real optimizer "
            "update, strict checkpoint reload, and one frozen-environment policy step."
        ),
    )
    model_smoke.add_argument(
        "--model", choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"), required=True
    )
    model_smoke.add_argument("--fixture", action="store_true", required=True)
    model_smoke.add_argument("--output", type=Path, required=True)
    model_smoke.add_argument(
        "--reload-as",
        choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"),
        help="adversarial trusted-identity probe; mismatches fail before environment creation",
    )

    bundle = commands.add_parser(
        "export-inference-bundle",
        description="Export any pinned checkpoint to a digest-bound tensor-only bundle.",
    )
    bundle.add_argument("--checkpoint", type=Path, required=True)
    bundle.add_argument("--config", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--artifact-id", required=True)
    bundle.add_argument("--artifact-index", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate-model",
        description="Evaluate one trusted bundle on ordered seeds 100000..100099.",
    )
    evaluate.add_argument("--model", choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"))
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--artifact-id", required=True)
    evaluate.add_argument("--artifact-index", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")

    compare = commands.add_parser(
        "compare-models",
        description=(
            "Generate deterministic JSON and Markdown from exactly one digest-locked "
            "anchored final evaluation per model."
        ),
    )
    compare.add_argument("--artifact-index", type=Path, required=True)
    compare.add_argument(
        "--artifact-id",
        dest="artifact_ids",
        action="append",
        required=True,
        help="repeat exactly four times, once per anchored final model evaluation",
    )
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--validate-existing",
        action="store_true",
        help="read-only validation for deterministic resume; never creates or changes reports",
    )

    resume = commands.add_parser(
        "validate-production-artifact",
        description="Read-only validation of a completed production stage before resume skip.",
    )
    resume.add_argument("--stage", choices=("training", "bundle", "evaluation"), required=True)
    resume.add_argument(
        "--model", choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"), required=True
    )
    resume.add_argument("--artifact-id", required=True)
    resume.add_argument("--artifact-index", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)

    smoke_env = commands.add_parser(
        "native-env-smoke",
        description="Reset, render both native cameras/state, step exact float32 XY, and close.",
    )
    smoke_env.add_argument("--seed", type=int, default=100000)
    smoke_env.add_argument("--steps", type=int, default=1)
    smoke_env.add_argument("--action", type=float, nargs="+", default=[0.0, 0.0])
    smoke_env.add_argument("--evidence", type=Path, required=True)
    return parser


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NativeCliError(f"malformed native {label} config: mapping required")
    return cast("dict[str, object]", value)


def _strict_equal(actual: object, expected: object) -> bool:
    """Compare parsed YAML recursively without Python's bool/int equality coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        actual_mapping = cast("dict[object, object]", actual)
        expected_mapping = cast("dict[object, object]", expected)
        return actual_mapping.keys() == expected_mapping.keys() and all(
            _strict_equal(actual_mapping[key], expected_value)
            for key, expected_value in expected_mapping.items()
        )
    if isinstance(expected, list):
        actual_list = cast("list[object]", actual)
        expected_list = cast("list[object]", expected)
        return len(actual_list) == len(expected_list) and all(
            _strict_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual_list, expected_list, strict=True)
        )
    return actual == expected


def _parse_exact_config(
    raw: dict[str, object], expected: ActiveConfig, role: ConfigRole
) -> ActiveConfig:
    if not _strict_equal(raw, expected):
        raise NativeCliError(f"malformed native {role} config contract")
    return cast("ActiveConfig", cast("object", raw))


def _load_active_config(role: ConfigRole, path: Path) -> ActiveConfig:
    policy = load_workspace_policy()
    expected_relative = policy["active_configs"][role]
    expected_path = PACKAGE_ROOT / expected_relative
    if path.resolve() != expected_path.resolve():
        raise NativeCliError(
            f"historical/inactive config rejected for {role}: {path}; active={expected_relative}"
        )
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise NativeCliError(f"cannot load native {role} config: {path}") from exc
    raw = _mapping(value, role)
    expected_config: ActiveConfig
    if role == "benchmark":
        expected_config = _BENCHMARK_EXPECTED
    elif role == "collection":
        expected_config = _COLLECTION_EXPECTED
    else:
        expected_config = _EXPORT_EXPECTED
    return _parse_exact_config(raw, expected_config, role)


def _validate_contract(config: Path) -> dict[str, object]:
    _load_active_config("benchmark", config)
    policy = load_workspace_policy()
    return {
        "identifier": "pusht_so100_native_v1",
        "plan": policy["workspace"]["plan"],
        "contract_schema": policy["native_contract"]["schema"],
        "observation": "cam_top:uint8[3,224,224];cam_side:uint8[3,224,224];agent_pos:float32[5]",
        "action": "absolute_mocap_xy:float32[2]:normalized",
        "fps": 10,
    }


def _export_preflight(config: Path) -> dict[str, object]:
    _load_active_config("export", config)
    result: dict[str, object] = dict(native_runtime_report())
    result.update(
        {
            "command": "export-native",
            "adapter": "frozen_pushT_so100",
            "config": str(config.resolve().relative_to(PACKAGE_ROOT.resolve())),
            "export_contract": "native_lerobot_identity",
        }
    )
    return result


def _import_native(repo: Path, output: Path, config: Path) -> int:
    _load_active_config("export", config)
    native_runtime_report()
    from .data.importer import import_repo_store

    return import_repo_store(repo, output)


def _export_native(root: Path, output: Path, config: Path) -> Path:
    _load_active_config("export", config)
    native_runtime_report()
    from .data.exporter import export_paper_view, runtime_lock_digest

    lock = PACKAGE_ROOT / "environments/sim-runtime.lock"
    try:
        return export_paper_view(root, output, runtime_lock_digest=runtime_lock_digest(lock))
    except ValueError as exc:
        raise NativeCliError(str(exc)) from exc


def _dry_run_manifest_plan(metadata_path: Path, experiment_config: Path) -> dict[str, object]:
    """Validate native collection progress and plan splits without touching runtime artifacts."""
    import hashlib

    from .data.splits import SplitError, build_split_manifest, load_experiment_config

    try:
        config = load_experiment_config(experiment_config)
    except SplitError as exc:
        raise NativeCliError(str(exc)) from exc
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise NativeCliError(f"native collection metadata is not a regular file: {metadata_path}")
    try:
        payload = metadata_path.read_bytes()
        value: object = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeCliError(f"cannot load native collection metadata: {metadata_path}") from exc
    if not isinstance(value, dict):
        raise NativeCliError("native collection metadata must be an object")
    metadata = cast("dict[str, object]", value)
    accepted = metadata.get("total_episodes")
    if type(accepted) is not int or accepted < 0:
        raise NativeCliError(
            "native collection metadata total_episodes must be a non-negative integer"
        )
    if metadata.get("codebase_version") != "v3.0" or metadata.get("fps") != 10:
        raise NativeCliError("native collection metadata must be LeRobot v3.0 at FPS 10")
    try:
        manifest = build_split_manifest(
            [str(index) for index in range(accepted)],
            config,
            source_digest=hashlib.sha256(payload).hexdigest(),
        )
    except SplitError as exc:
        raise NativeCliError(str(exc)) from exc
    return {
        "status": "dry-run-manifest-planned",
        "artifacts_created": False,
        "accepted_episode_count": accepted,
        "target_episode_count": manifest.target_episode_count,
        "remaining_episode_count": 0,
        "selected_episode_ids": list(manifest.selected_episode_ids),
        "split_counts": {
            name: len(manifest.members(name)) for name in ("train", "validation", "test")
        },
        "split_digest": manifest.digest,
    }


def _freeze_experiment(
    source: Path, output: Path, experiment_config: Path, sessions_path: Path | None
) -> dict[str, object]:
    native_runtime_report()
    from .data.paper_view import PaperViewError
    from .data.splits import SplitError, freeze_training_view, load_experiment_config

    try:
        config = load_experiment_config(experiment_config)
    except SplitError as exc:
        raise NativeCliError(str(exc)) from exc
    sessions: dict[str, str] | None = None
    if sessions_path is not None:
        try:
            value: object = json.loads(sessions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NativeCliError(f"cannot load session metadata: {sessions_path}") from exc
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in cast("dict[object, object]", value).items()
        ):
            raise NativeCliError("session metadata must map episode IDs to session IDs")
        sessions = cast("dict[str, str]", value)
    try:
        destination, manifest = freeze_training_view(source, output, config, sessions=sessions)
    except (SplitError, PaperViewError) as exc:
        raise NativeCliError(str(exc)) from exc
    return {
        "status": "frozen",
        "output": str(destination),
        "target_episode_count": manifest.target_episode_count,
        "selected_episode_ids": list(manifest.selected_episode_ids),
        "split_counts": {
            name: len(manifest.members(name)) for name in ("train", "validation", "test")
        },
        "split_digest": manifest.digest,
    }


def _collection_preflight(args: argparse.Namespace) -> NativeCollectionPlan:
    _load_active_config("benchmark", args.config)
    _load_active_config("collection", args.collection_config)
    return preflight_native_collection(
        NativeCollectionRequest(
            dataset_root=args.dataset_root,
            fps=args.fps,
            move_speed=args.move_speed,
            rotation_speed=args.rotation_speed,
        ),
        runtime_report=native_runtime_report,
    )


def _train_model(args: argparse.Namespace) -> dict[str, object]:
    """Dispatch a non-final smoke proof or a bounded full-production training run."""
    raw_max_updates: object = args.max_updates
    max_updates = raw_max_updates if type(raw_max_updates) is int else None
    expected_updates: int | None = None
    if args.full_production:
        if args.paper_profiles is None:
            raise NativeCliError("--full-production requires --paper-profiles")
        from .training.paper_profiles import load_paper_profiles

        expected_updates = load_paper_profiles(args.paper_profiles).models[
            args.model
        ].resolved_optimizer_updates
        if max_updates != expected_updates:
            raise NativeCliError(
                f"--full-production requires --max-updates {expected_updates} for {args.model}"
            )
    if not args.full_production and raw_max_updates is not None:
        raise NativeCliError("--max-updates is valid only with --full-production")
    try:
        from .training.artifacts import ArtifactIndex
        from .training.launcher import (
            TrainingLaunch,
            launch_training,
            preflight_full_production,
        )

        if args.preflight:
            if not args.full_production:
                raise NativeCliError("train-model --preflight requires --full-production")
            if max_updates is None:
                raise NativeCliError("--full-production requires the selected paper budget")
            return preflight_full_production(args.paper_view, args.model, args.seed, max_updates)
        artifact_root = PROJECT_ROOT / "04_experiments/so101_pusht_benchmark"
        index = ArtifactIndex(args.artifact_index, artifact_root)
        smoke_mode = args.smoke_mode or ("fixture" if args.smoke else None)
        training_mode = "full_production" if args.full_production else None
        checkpoint = launch_training(
            args.paper_view,
            args.output,
            index,
            TrainingLaunch(
                args.seed,
                args.artifact_id,
                model=args.model,
                smoke=args.smoke,
                smoke_mode=None if args.smoke else smoke_mode,
                training_mode=training_mode,
                max_updates=max_updates,
            ),
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc
    return {
        "checkpoint": str(checkpoint),
        "model": args.model,
        "smoke_mode": smoke_mode,
        "training_mode": training_mode,
        "configured_optimizer_updates": max_updates if training_mode else 1,
        "artifact": index.record(args.artifact_id),
    }


def _model_smoke(args: argparse.Namespace) -> Path:
    try:
        from .training.vertical_smoke import run_model_smoke

        return run_model_smoke(
            args.model,
            args.output,
            fixture=args.fixture,
            reload_model=args.reload_as,
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc


def _artifact_index(path: Path) -> ArtifactIndex:
    return ArtifactIndex(path, PROJECT_ROOT / "04_experiments/so101_pusht_benchmark")


def _export_bundle(args: argparse.Namespace) -> Path:
    try:
        from .training.exporter import export_inference_bundle

        return export_inference_bundle(
            args.checkpoint,
            args.config,
            args.output,
            artifact_id=args.artifact_id,
            index=_artifact_index(args.artifact_index),
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc


def _evaluate_model(args: argparse.Namespace) -> Path:
    try:
        native_runtime_report()
        from .training.evaluator import EvaluationRequest, evaluate_bundle

        return evaluate_bundle(
            args.bundle,
            args.output,
            _artifact_index(args.artifact_index),
            EvaluationRequest(args.artifact_id, model=args.model, device=args.device),
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc


def _compare_models(args: argparse.Namespace) -> dict[str, object]:
    try:
        from .evaluation.comparative_report import (
            validate_existing_comparative_report_from_index,
            write_comparative_report_from_index,
        )

        operation = (
            validate_existing_comparative_report_from_index
            if args.validate_existing
            else write_comparative_report_from_index
        )
        json_path, markdown_path = operation(
            _artifact_index(args.artifact_index), tuple(args.artifact_ids), args.output
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc
    return {
        "status": "reused" if args.validate_existing else "generated",
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def _validate_production_artifact(args: argparse.Namespace) -> dict[str, object]:
    try:
        from .training import validate_production_resume_artifact

        return validate_production_resume_artifact(
            _artifact_index(args.artifact_index),
            stage=args.stage,
            model=args.model,
            artifact_id=args.artifact_id,
            output=args.output,
        )
    except Exception as exc:
        raise NativeCliError(str(exc)) from exc


def _native_env_smoke(args: argparse.Namespace) -> dict[str, object]:
    import hashlib
    import shutil

    import imageio.v3 as iio
    import numpy as np

    from .evaluation.frozen_env import ActionContractError, load_frozen_pusht, validate_action

    if type(args.steps) is not int or args.steps < 1 or args.steps > 300:
        raise NativeCliError("native env smoke steps must be in 1..300")
    raw_action: object = args.action
    if not isinstance(raw_action, list):
        raise NativeCliError("action must be exact float32[2]")
    action_values = cast("list[object]", raw_action)
    if len(action_values) != 2:
        raise NativeCliError("action must be exact float32[2]")
    for value in action_values:
        if type(value) not in (int, float):
            raise NativeCliError("action must contain numeric values; bool is forbidden")
        number = cast("int | float", value)
        if isinstance(number, float) and not math.isfinite(number):
            raise NativeCliError("action must contain only finite values")
        if number < -1.0 or number > 1.0:
            raise NativeCliError("action must remain within [-1,1] bounds; clipping is forbidden")
    try:
        checked = validate_action(np.asarray(action_values, dtype=np.float32))
    except ActionContractError as exc:
        raise NativeCliError(str(exc)) from exc
    evidence = args.evidence.resolve(strict=False)
    if not evidence.parent.is_dir() or evidence.exists() or evidence.is_symlink():
        raise NativeCliError(
            "native env evidence destination must be absent with an existing parent"
        )
    token = hashlib.sha256(str(evidence).encode()).hexdigest()[:12]
    staging = evidence.with_name(f".{evidence.name}.tmp-{token}")
    if staging.exists() or staging.is_symlink():
        raise NativeCliError("native env evidence staging already exists")
    native_runtime_report()
    environment = load_frozen_pusht(max_steps=args.steps)
    receipt: dict[str, object]
    try:
        observation, _ = environment.reset(seed=args.seed)
        final = None
        for _ in range(args.steps):
            final = environment.step(checked)
        staging.mkdir()
        iio.imwrite(staging / "cam_top.png", observation["cam_top"], fps=10)
        iio.imwrite(staging / "cam_side.png", observation["cam_side"], fps=10)
        receipt = {
            "schema": 1,
            "seed": args.seed,
            "requested_steps": args.steps,
            "environment_steps": args.steps,
            "closed": True,
            "cam_top": {"shape": [224, 224, 3], "dtype": "uint8"},
            "cam_side": {"shape": [224, 224, 3], "dtype": "uint8"},
            "agent_pos": {
                "shape": [5],
                "dtype": "float32",
                "value": observation["agent_pos"].tolist(),
            },
            "action": checked.tolist(),
            "terminated": final.terminated if final is not None else False,
            "truncated": final.truncated if final is not None else False,
            "dxy": final.info["dxy"] if final is not None else None,
            "dyaw": final.info["dyaw"] if final is not None else None,
        }
        (staging / "smoke.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        environment.close()
    try:
        staging.replace(evidence)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "verified", "evidence": str(evidence), **receipt}


def main(argv: list[str] | None = None) -> int:
    """Run the active native parser; historical commands are intentionally absent."""
    executable = Path(sys.argv[0]).name
    active_commands = {
        "validate-contract",
        "inspect-env",
        "collect-native",
        "import-native",
        "export-native",
        "freeze-experiment",
        "train-model",
        "model-smoke",
        "export-inference-bundle",
        "evaluate-model",
        "compare-models",
        "validate-production-artifact",
        "native-env-smoke",
    }
    if argv is None and executable in active_commands:
        argv = [executable, *sys.argv[1:]]
    args = command_parser().parse_args(argv)
    try:
        if args.command == "validate-contract":
            contract = _validate_contract(args.config)
            output = (
                f"contract.identifier={contract['identifier']}\n"
                f"contract.plan={contract['plan']}\n"
                f"contract.schema={contract['contract_schema']}\n"
                f"contract.observation={contract['observation']}\n"
                f"contract.action={contract['action']}\n"
                f"contract.timing={contract['fps']}Hz;timestamp=frame_index/10"
            )
        elif args.command == "inspect-env":
            output = json.dumps(native_runtime_report(), sort_keys=True)
        elif args.command == "collect-native":
            plan = _collection_preflight(args)
            report = plan.report()
            report["config"] = str(args.config.resolve().relative_to(PACKAGE_ROOT.resolve()))
            report["collection_config"] = str(
                args.collection_config.resolve().relative_to(PACKAGE_ROOT.resolve())
            )
            if args.launch:
                if DEVICE_PROBE_ENV in os.environ or DISPLAY_PROBE_ENV in os.environ:
                    raise NativeCollectionError("injected probes are preflight-only")
                returncode = launch_native_collection(plan)
                if returncode != 0:
                    print(f"FAIL CLOSED: native collector exited with status {returncode}")
                    return returncode if returncode > 0 else 1
                report["status"] = "completed"
                report["returncode"] = 0
            output = json.dumps(report, sort_keys=True)
        elif args.command == "import-native":
            return _import_native(args.repo, args.output, args.config)
        elif args.command == "freeze-experiment":
            if args.metadata is not None:
                if not args.dry_run:
                    raise NativeCliError("--metadata requires --dry-run")
                if args.output is not None or args.sessions is not None:
                    raise NativeCliError("metadata dry-run does not accept --output or --sessions")
                result = _dry_run_manifest_plan(args.metadata, args.experiment_config)
            else:
                if args.dry_run:
                    raise NativeCliError("--dry-run requires --metadata")
                if args.source is None or args.output is None:
                    raise NativeCliError("freeze-experiment with --source requires --output")
                result = _freeze_experiment(
                    args.source, args.output, args.experiment_config, args.sessions
                )
            output = json.dumps(result, sort_keys=True)
        elif args.command == "train-model":
            output = json.dumps(_train_model(args), sort_keys=True)
        elif args.command == "model-smoke":
            output = json.dumps({"result": str(_model_smoke(args))}, sort_keys=True)
        elif args.command == "export-inference-bundle":
            output = json.dumps({"bundle": str(_export_bundle(args))}, sort_keys=True)
        elif args.command == "evaluate-model":
            output = json.dumps({"metrics": str(_evaluate_model(args))}, sort_keys=True)
        elif args.command == "compare-models":
            output = json.dumps(_compare_models(args), sort_keys=True)
        elif args.command == "validate-production-artifact":
            output = json.dumps(_validate_production_artifact(args), sort_keys=True)
        elif args.command == "native-env-smoke":
            output = json.dumps(_native_env_smoke(args), sort_keys=True)
        elif args.preflight:
            if args.root is not None or args.output is not None:
                raise NativeCliError("export preflight does not accept data paths")
            output = json.dumps(_export_preflight(args.config), sort_keys=True)
        else:
            if args.root is None or args.output is None:
                raise NativeCliError("export-native requires --root and --output")
            destination = _export_native(args.root, args.output, args.config)
            output = json.dumps({"exported": str(destination)}, sort_keys=True)
    except (NativeCliError, NativeCollectionError, NativeRuntimeError, WorkspacePolicyError) as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
