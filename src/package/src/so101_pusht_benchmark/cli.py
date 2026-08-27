"""Contract, simulator, and local-only gamepad collection command-line surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .integrations.lerobot.gamepad import GamepadSample


class _MousePaperState:
    """Paper-state adapter from the env's live pose tuple."""

    def __init__(
        self, t_x: float, t_y: float, t_yaw: float, pusher_x: float, pusher_y: float
    ) -> None:
        self._t_x = t_x
        self._t_y = t_y
        self._t_yaw = t_yaw
        self._pusher_x = pusher_x
        self._pusher_y = pusher_y

    @property
    def t_x(self) -> float:
        return self._t_x

    @property
    def t_y(self) -> float:
        return self._t_y

    @property
    def t_yaw(self) -> float:
        return self._t_yaw

    @property
    def pusher_x(self) -> float:
        return self._pusher_x

    @property
    def pusher_y(self) -> float:
        return self._pusher_y


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/benchmark/pusht_v1.yaml"
_COLLECTION_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/collection/pusht_gamepad_v1.yaml"
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_CONFIG = _PACKAGE_ROOT / "configs/export/paper_view_v1.yaml"
_ARTIFACT_ROOT = Path(__file__).resolve().parents[4] / "04_experiments/so101_pusht_benchmark"
_ARTIFACT = _ARTIFACT_ROOT / "datasets"
_ARTIFACT_INDEX = _PACKAGE_ROOT / "configs/provenance/artifact_index.json"


class _Clock:
    def monotonic(self) -> float:
        return time.monotonic()


class _Synthetic:
    def poll(self) -> GamepadSample:
        from .integrations.lerobot.gamepad import GamepadSample

        return GamepadSample((0.0, 0.0, 0.0), True)

    def close(self) -> None:
        pass


def command_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="so101-pusht-benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate-contract")
    v.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    v.add_argument("--require-safety-ready", action="store_true")
    for name in ("inspect-env", "validate-sim", "render", "step-smoke", "calibrate-sim"):
        q = sub.add_parser(name)
        q.add_argument("--seed", type=int, default=0)
        if name == "step-smoke":
            q.add_argument("--probe", choices=("happy", "malformed"), default="happy")
    probe = sub.add_parser("probe-gamepad")
    probe.add_argument("--json", action="store_true")
    collect = sub.add_parser("collect-sim")
    collect.add_argument("--root", type=Path, default=_ARTIFACT / "human_pilot")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--attempt-id", default="attempt_0")
    collect.add_argument("--synthetic-pipeline-probe", action="store_true")
    collect.add_argument("--ticks", type=int, default=300)
    # 0 (default) keeps the session open across takes until an operator stop;
    # a positive value caps the number of takes per launch.
    collect.add_argument("--max-attempts", type=int, default=0)
    collect.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    collect.add_argument("--collection-config", type=Path, default=_COLLECTION_CONFIG)
    pilot = sub.add_parser("validate-pilot")
    pilot.add_argument("--root", type=Path, default=_ARTIFACT / "human_pilot")
    replay = sub.add_parser("replay-episode")
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--attempt-id", required=True)
    export = sub.add_parser("export-paper-view")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--output", type=Path, default=_ARTIFACT)
    export.add_argument("--config", type=Path, default=_EXPORT_CONFIG)
    train = sub.add_parser("train-model")
    train.add_argument(
        "--model",
        choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm"),
        default="dp_cnn",
    )
    train.add_argument("--paper-view", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    train.add_argument("--artifact-id", required=True)
    train.add_argument("--artifact-index", type=Path, default=_ARTIFACT_INDEX)
    train.add_argument("--synthetic-pipeline-probe", action="store_true")
    train.add_argument("--smoke", action="store_true", help="200-step bounded smoke run")
    bundle = sub.add_parser("export-inference-bundle")
    bundle.add_argument("--checkpoint", type=Path, required=True)
    bundle.add_argument("--config", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--artifact-id", required=True)
    bundle.add_argument("--artifact-index", type=Path, default=_ARTIFACT_INDEX)
    evaluate = sub.add_parser("evaluate-model")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=_ARTIFACT_ROOT / "evaluations/dp_cnn")
    evaluate.add_argument("--artifact-id", default="dp_cnn-seed0")
    evaluate.add_argument("--artifact-index", type=Path, default=_ARTIFACT_INDEX)
    import_repo = sub.add_parser("import-repo-store")
    import_repo.add_argument("--repo", type=Path, required=True)
    import_repo.add_argument("--output", type=Path, required=True)
    return p


def _validate_contract(config: Path, require_safety_ready: bool) -> int:
    from .task.spec import TaskSpec, TaskSpecError

    try:
        spec = TaskSpec.from_yaml(config)
        if require_safety_ready:
            spec.require_safety_ready()
    except (OSError, TaskSpecError) as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    print(
        f"contract.identifier={spec.identifier}\ncontract.observation=front:uint8[96,96,3];state:float32[15]:radians+radians_per_second+metres\ncontract.action=absolute_ee_xyz:float32[3]:meters\ncontract.timing={spec.policy_fps}Hz;dt={spec.mujoco_dt};substeps={spec.substeps}\ncontract.models=DP-CNN:2/16/8;DP-Transformer:2/16/8;IBC:2/2/1;LSTM-GMM:10/10/1\ncontract.deployment_scope={spec.deployment_scope}\ncontract.safety_ready={require_safety_ready}"
    )
    return 0


def _sim(command: str, seed: int, probe: str = "happy") -> int:
    from .sim.env import PushTEnv
    import numpy as np

    env = PushTEnv()
    try:
        obs, _ = env.reset(seed)
        if command == "calibrate-sim":
            from .sim.calibration import calibrate

            print(json.dumps({"evidence": str(calibrate(seed))}, sort_keys=True))
            return 0
        print(
            f"reset.seed={seed} rgb.sha256={hashlib.sha256(obs['observation.images.front'].tobytes()).hexdigest()}"
        )
        if command in ("step-smoke", "validate-sim"):
            before_qpos = env.scene.data.qpos.tobytes()
            before_time = float(env.scene.data.time)
            out = env.step(
                np.asarray([0.30, 0.0, 0.050], dtype=np.float32)
                if probe == "happy"
                else np.asarray([float("nan"), 0.0, 0.050], dtype=np.float32)
            )
            evidence = {
                "probe": probe,
                "fault": out.info.get("fault"),
                "timestamp": out.info.get("timestamp"),
                "qpos_unchanged": before_qpos == env.scene.data.qpos.tobytes(),
                "time_unchanged": before_time == float(env.scene.data.time),
            }
            print(json.dumps(evidence, sort_keys=True))
            return int(bool(out.info.get("fault")) != (probe == "malformed"))
        return 0
    finally:
        env.close()


def _collect(args: argparse.Namespace) -> int:
    import numpy as np

    from .collection.inputs import CollectionInput
    from .collection.recorder import CollectionConfig, PollingSource, Recorder
    from .collection.types import MouseCollectionConfig
    from .collection.viewer import LiveViewer, OverlayState, RealtimePacer
    from .control.paper_view import PaperView as ControlPaperView
    from .data.store import LocalDatasetStore
    from .input.mouse_keyboard import MouseKeyboardSource
    from .integrations.lerobot.gamepad import PublicGamepadSource
    from .sim.env import PushTEnv
    from .task.spec import TaskSpec, TaskSpecError

    try:
        spec = TaskSpec.from_yaml(args.config)
        is_schema_3 = spec.schema == 3
        if is_schema_3:
            mouse_config = MouseCollectionConfig.load(args.collection_config)
            collection_config = None
        else:
            mouse_config = None
            collection_config = CollectionConfig.load(args.collection_config)
    except (OSError, TaskSpecError) as exc:
        print(json.dumps({"accepted": False, "failure_code": "config_error", "detail": str(exc)}))
        return 1

    viewer = LiveViewer.open(
        enabled=not args.synthetic_pipeline_probe and bool(os.environ.get("DISPLAY")),
        title="PushT control (paper topdown)",
    )
    observer: LiveViewer | None = None

    try:
        source: PollingSource
        if args.synthetic_pipeline_probe:
            source = _Synthetic()
        elif is_schema_3:
            if viewer.root is None:
                raise RuntimeError("no display for mouse keyboard source")
            assert mouse_config is not None
            source = MouseKeyboardSource(
                viewer.root,
                mouse_config.bounds_x,
                mouse_config.bounds_y,
            )
            # Separate observer window for the 3D robot scene; it never
            # receives control input, so clicks there cannot fault attempts.
            observer = LiveViewer.open(
                enabled=bool(os.environ.get("DISPLAY")),
                title="SO-101 3D observer",
            )
        else:
            source = PublicGamepadSource()
    except (ImportError, RuntimeError, OSError) as exc:
        viewer.close()
        if observer is not None:
            observer.close()
        print(json.dumps({"accepted": False, "failure_code": "no_device", "detail": str(exc)}))
        return 1
    _ARTIFACT.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        Path(tempfile.mkdtemp(prefix="synthetic-pipeline-", dir=_ARTIFACT))
        if args.synthetic_pipeline_probe
        else args.root
    )
    env = None
    try:
        store = LocalDatasetStore(temporary_root)
        env = PushTEnv()
        clock = _Clock()
        if is_schema_3 and mouse_config is not None:
            recorder = Recorder(
                env,
                source,
                store,
                CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050),
                clock,
                input_adapter=CollectionInput.mouse(
                    stale_timeout_s=mouse_config.stale_timeout_s,
                    debounce_ticks=mouse_config.debounce_ticks,
                    contact_z_m=mouse_config.contact_z_m,
                    clearance_z_m=mouse_config.clearance_z_m,
                    bounds_x=mouse_config.bounds_x,
                    bounds_y=mouse_config.bounds_y,
                ),
            )
        else:
            assert collection_config is not None
            recorder = Recorder(env, source, store, collection_config, clock)

        if args.synthetic_pipeline_probe:
            mode = "synthetic_pipeline_probe"
            on_obs = None
        elif is_schema_3:
            mode = "human_mouse_keyboard"
            assert mouse_config is not None
            bounds_x, bounds_y = mouse_config.bounds_x, mouse_config.bounds_y

            def on_obs_mouse(observation: object) -> None:
                # Control pane: canonical PushT-style 2D paper view; observer
                # pane: full 3D robot scene. The raw 96x96 policy topdown
                # stays in the recorder's observation.
                obs = cast("dict[str, object]", observation)
                tx, ty, tyaw, px, py = env.paper_state
                paper = ControlPaperView(bounds_x=bounds_x, bounds_y=bounds_y).render(
                    _MousePaperState(tx, ty, tyaw, px, py), size=384
                )
                display_front = env.scene.render(camera="front", size=384, hide_robot=False)
                state = cast(
                    "np.ndarray[tuple[int, ...], np.dtype[np.float32]]",
                    obs["observation.state"],
                )
                if isinstance(source, MouseKeyboardSource):
                    state_str = "ARMED" if source.deadman else "READY"
                    if not source.has_focus:
                        state_str = "NO FOCUS"
                    if not source.connected:
                        state_str = "DISCONNECTED"
                    overlay = OverlayState(
                        measured_ee=(float(state[12]), float(state[13])),
                        z_level=source.current_z,
                        state=state_str,
                    )
                else:
                    overlay = None
                viewer.show(
                    paper,
                    overlay=overlay,
                    bounds_x=bounds_x,
                    bounds_y=bounds_y,
                )
                if observer is not None:
                    observer.show(display_front)

            on_obs = on_obs_mouse
        else:
            mode = "human_gamepad"

            def on_obs_gamepad(observation: object) -> None:
                frame = cast("np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", observation)
                viewer.show(frame)

            on_obs = on_obs_gamepad

        # The session keeps the windows open across attempts: each deadman
        # release or fault ends one take, then the next press starts a new
        # one. Only an operator stop (Esc), a keyboard interrupt, or the
        # attempt cap ends the whole session.
        max_attempts = args.max_attempts if args.max_attempts > 0 else 10**9
        session_exit = 0
        for attempt_no in range(max_attempts):
            attempt_id = args.attempt_id if attempt_no == 0 else f"{args.attempt_id}_{attempt_no}"
            result = recorder.record(
                args.seed + attempt_no,
                attempt_id,
                _mode=mode,
                max_ticks=args.ticks,
                before_tick=(None if args.synthetic_pipeline_probe else RealtimePacer(clock).wait),
                on_observation=on_obs,
            )
            if args.synthetic_pipeline_probe:
                result_json = {
                    "mode": mode,
                    "success": False,
                    "training_eligible": False,
                    "qa": "raw-attempt probe completed and temporary storage will be removed",
                }
            else:
                result_json = {
                    "accepted": result.accepted,
                    "failure_code": result.failure_code,
                    "frames": result.frames,
                    "attempt": str(result.attempt_path),
                }
            print(json.dumps(result_json, sort_keys=True))
            # The probe runs exactly one take; an operator stop (Esc) ends the
            # session; other faults (deadman release, invalid target, coverage)
            # start the next take.
            if args.synthetic_pipeline_probe:
                break
            if result.failure_code in ("software_stop", "operator_success_unverified"):
                session_exit = 0
                break
            if attempt_no + 1 >= max_attempts:
                break
        # A completed human attempt, including an operator stop, disconnect, or
        # interrupt, is a persisted collection result rather than a CLI error.
        return session_exit
    finally:
        viewer.close()
        if observer is not None:
            observer.close()
        source.close()
        if env is not None:
            env.close()
        if args.synthetic_pipeline_probe:
            shutil.rmtree(temporary_root)


def _export_paper_view(root: Path, output: Path, config: Path) -> int:
    import yaml

    from .data.exporter import ExportError, export_paper_view, runtime_lock_digest
    from .data.paper_view import EXPORTER_REVISION, PaperViewError
    from .workspace import WorkspacePolicyError

    try:
        value: object = yaml.safe_load(config.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ExportError("export config must be an object")
        raw = cast("dict[str, object]", value)
        zarr = raw.get("zarr")
        if (
            raw.get("schema") != 1
            or raw.get("exporter_revision") != EXPORTER_REVISION
            or not isinstance(zarr, dict)
            or cast("dict[str, object]", zarr)
            != {
                "format": 2,
                "order": "C",
                "compressor": None,
                "filters": None,
            }
        ):
            raise ExportError("export config contract mismatch")
        lock_value = raw.get("runtime_lock")
        if (
            not isinstance(lock_value, str)
            or Path(lock_value).is_absolute()
            or ".." in Path(lock_value).parts
        ):
            raise ExportError("unsafe runtime lock config path")
        lock = (_PACKAGE_ROOT / lock_value).resolve()
        if _PACKAGE_ROOT not in lock.parents:
            raise ExportError("runtime lock escapes package root")
        view = export_paper_view(
            root,
            output,
            runtime_lock_digest=runtime_lock_digest(lock),
        )
    except (OSError, yaml.YAMLError, ExportError, PaperViewError, WorkspacePolicyError) as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    print(json.dumps({"paper_view": str(view)}, sort_keys=True))
    return 0


def _model_pipeline(args: argparse.Namespace) -> int:
    try:
        from .training.artifacts import ArtifactIndex

        index = ArtifactIndex(args.artifact_index, _ARTIFACT_ROOT)
        if args.command == "train-model":
            from .training.launcher import TrainingLaunch, launch_training

            result = launch_training(
                args.paper_view,
                args.output,
                index,
                TrainingLaunch(
                    args.seed,
                    args.artifact_id,
                    args.synthetic_pipeline_probe,
                    model=args.model,
                    smoke=args.smoke,
                ),
            )
        elif args.command == "export-inference-bundle":
            from .training.exporter import export_inference_bundle

            result = export_inference_bundle(
                args.checkpoint,
                args.config,
                args.output,
                artifact_id=args.artifact_id,
                index=index,
            )
        else:
            from .training.evaluator import EvaluationRequest, evaluate_bundle

            result = evaluate_bundle(
                args.bundle,
                args.output,
                index,
                EvaluationRequest(args.artifact_id),
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FAIL CLOSED: {exc}")
        return 1
    print(json.dumps({"artifact": str(result)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    executable = Path(sys.argv[0]).name
    commands = {
        "validate-contract",
        "inspect-env",
        "validate-sim",
        "step-smoke",
        "calibrate-sim",
        "probe-gamepad",
        "collect-sim",
        "validate-pilot",
        "replay-episode",
        "export-paper-view",
        "train-model",
        "export-inference-bundle",
        "evaluate-model",
        "import-repo-store",
    }
    if argv is None and executable in commands:
        argv = [executable, *sys.argv[1:]]
    elif argv is None and executable == "render-sim":
        argv = ["render", *sys.argv[1:]]
    args = command_parser().parse_args(argv)
    if args.command == "validate-contract":
        return _validate_contract(args.config, args.require_safety_ready)
    if args.command in {"inspect-env", "validate-sim", "render", "step-smoke", "calibrate-sim"}:
        return _sim(args.command, args.seed, getattr(args, "probe", "happy"))
    if args.command == "probe-gamepad":
        from .integrations.lerobot.gamepad import PublicGamepadSource

        try:
            if args.json:
                from contextlib import redirect_stdout
                from io import StringIO

                with redirect_stdout(StringIO()):
                    source = PublicGamepadSource()
                    sample = source.poll()
            else:
                source = PublicGamepadSource()
                sample = source.poll()
        except (ImportError, RuntimeError, OSError) as exc:
            print(f"NO GAMEPAD: {exc}")
            return 1
        try:
            print(
                json.dumps(
                    {
                        "connected": sample.connected,
                        "fresh": sample.fresh,
                        "axes": sample.axes,
                        "deadman": sample.deadman,
                    },
                    sort_keys=True,
                )
            )
            return 0 if sample.connected else 1
        finally:
            source.close()
    if args.command == "collect-sim":
        return _collect(args)
    if args.command == "validate-pilot":
        from .data.validator import validate_pilot

        print(json.dumps(validate_pilot(args.root), sort_keys=True))
        return 0
    if args.command == "export-paper-view":
        return _export_paper_view(args.root, args.output, args.config)
    if args.command in {"train-model", "export-inference-bundle", "evaluate-model"}:
        return _model_pipeline(args)
    if args.command == "import-repo-store":
        from .data.importer import import_repo_store

        return import_repo_store(args.repo, args.output)
    from .data.validator import replay_attempt

    print(json.dumps(replay_attempt(args.root, args.attempt_id), sort_keys=True))
    return 0


def _native_module_main() -> int:
    try:
        from .native_cli import main as native_main
    except ModuleNotFoundError as exc:
        print(
            "FAIL CLOSED: native pushT-so100 runtime mismatch (fallback forbidden): "
            f"required module {exc.name or 'unknown'} is missing"
        )
        return 1
    return native_main()


if __name__ == "__main__":
    raise SystemExit(_native_module_main())
