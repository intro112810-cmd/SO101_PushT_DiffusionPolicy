"""Create one DP-Transformer 3D video and analysis figures for feedback."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import cv2
import imageio.v2 as iio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
from so101_pusht_benchmark.integrations.paper_baselines.runner import (
    PaperBaselineRunner,
    _preserve_policy_rng,
    _seed_policy_rng,
    validate_native_runner_observation,
)
from so101_pusht_benchmark.training.artifacts import ArtifactIndex, sha256_file
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity
from so101_pusht_benchmark.training.metadata import (
    read_normalizer_metadata,
    read_trusted_config,
)


@dataclass(frozen=True, slots=True)
class Capture:
    label: str
    seed: int
    success: bool
    frames: list[np.ndarray]
    top_frames: list[np.ndarray]
    targets: list[np.ndarray]
    blocks: list[np.ndarray]
    goal: np.ndarray
    end_effectors: list[np.ndarray]
    block_yaw: float
    goal_yaw: float


@dataclass(frozen=True, slots=True)
class CaptureOptions:
    render: bool = True
    policy_seed: int | None = None
    max_steps: int = 300


_DEFAULT_CAPTURE_OPTIONS = CaptureOptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--artifact", default="local-dp_transformer-seed0")
    parser.add_argument("--model", default="dp_transformer")
    parser.add_argument("--success-seed", default=100006, type=int)
    parser.add_argument("--failure-seed", default=100008, type=int)
    return parser.parse_args()


def load_policy(
    root: Path, artifact_id: str, model: str
) -> tuple[torch.nn.Module, BundleIdentity]:
    index = ArtifactIndex(root / "artifact-index.json", root)
    record = index.record(artifact_id)
    identity = BundleIdentity.from_dict(record.get("identity"))
    if identity.model != model:
        raise RuntimeError(f"artifact {identity.model} != requested {model}")
    checkpoint = index.verify(artifact_id, "checkpoint")
    config_path = index.verify(artifact_id, "config")
    normalizer = index.verify(artifact_id, "normalizer")
    bundle = index.verify(artifact_id, "bundle")
    checkpoint_digest = sha256_file(checkpoint)
    config_digest = sha256_file(config_path)
    config = read_trusted_config(config_path, identity.model)
    policy = instantiate(OmegaConf.create(config["policy"]))
    expected = dict(policy.state_dict())
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    for key, (shape, dtype_name) in read_normalizer_metadata(
        normalizer, identity, checkpoint_digest, config_digest
    ).items():
        expected[key] = torch.empty(shape, dtype=dtypes[dtype_name])
    state = load_bundle(
        bundle,
        expected,
        index=index,
        artifact_id=artifact_id,
        expectation=BundleExpectation(identity, checkpoint_digest),
    )
    policy.load_state_dict(state, strict=True)
    policy.to("cuda:0")
    policy.eval()
    return policy, identity


def observation(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return validate_native_runner_observation(
        {key: value for key, value in raw.items() if not key.startswith("_")}
    )


def render_frame(
    renderer: mujoco.Renderer, data: mujoco.MjData, label: str, seed: int, step: int
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.23, 0.0, 0.08)
    camera.distance = 1.15
    camera.azimuth = 145
    camera.elevation = -28
    renderer.update_scene(data, camera=camera)
    frame = cv2.resize(renderer.render(), (1280, 720), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(frame, (20, 18), (500, 105), (8, 16, 28), -1)
    cv2.putText(frame, label, (36, 53), 0, 0.82, (255, 255, 255), 2)
    cv2.putText(frame, f"seed {seed}  |  step {step}", (36, 87), 0, 0.60, (160, 220, 255), 2)
    return frame


def capture_rollout(
    policy: torch.nn.Module,
    identity: BundleIdentity,
    seed: int,
    label: str,
    options: CaptureOptions = _DEFAULT_CAPTURE_OPTIONS,
) -> Capture:
    env = load_frozen_pusht(max_steps=options.max_steps)
    raw_env = env.raw_environment
    renderer = (
        mujoco.Renderer(raw_env.model, height=360, width=640)
        if options.render
        else None
    )
    runner = PaperBaselineRunner(
        Path(tempfile.gettempdir()) / "pusht-feedback-rollout",
        evaluation_seeds=(),
        n_obs_steps=identity.observation_steps,
        n_action_steps=identity.executed_actions,
        options={"max_steps": options.max_steps, "native_env_factory": "frozen"},
    )
    try:
        raw_observation, _ = env.reset(seed=seed)
        current = observation(raw_observation)
        top_frames = [np.asarray(raw_observation["_cam_top_hd"]).copy()]
        history = deque(
            (current for _ in range(identity.observation_steps)),
            maxlen=identity.observation_steps,
        )
        goal_site = raw_env.model.site("T_sign_anchor").id
        block_body = raw_env.model.body("T_block").id
        goal_body = raw_env.model.body("T_sign").id
        end_effector_body = raw_env.model.body("Fixed_Jaw").id
        goal = raw_env.data.site_xpos[goal_site, :2].copy()
        targets = [raw_env.data.mocap_pos[raw_env.mocap_id, :2].copy()]
        blocks = [raw_env.data.xpos[block_body, :2].copy()]
        end_effectors = [raw_env.data.xpos[end_effector_body, :2].copy()]
        block_yaw = float(
            np.arctan2(
                raw_env.data.xmat[block_body, 3],
                raw_env.data.xmat[block_body, 0],
            )
        )
        goal_yaw = float(
            np.arctan2(
                raw_env.data.xmat[goal_body, 3],
                raw_env.data.xmat[goal_body, 0],
            )
        )
        frames = [render_frame(renderer, raw_env.data, label, seed, 0)] if renderer else []
        terminated = False
        truncated = False
        steps = 0
        with _preserve_policy_rng(policy):
            _seed_policy_rng(
                policy,
                seed if options.policy_seed is None else options.policy_seed,
            )
            policy.reset()
            with torch.no_grad():
                while steps < options.max_steps and not terminated and not truncated:
                    prediction = policy.predict_action(runner.policy_observation(history, policy))
                    actions = prediction["action"].detach().cpu().numpy()
                    for value in actions[0, : identity.executed_actions]:
                        result = env.step(np.asarray(value, dtype=np.float32))
                        steps += 1
                        current = observation(result.observation)
                        top_frames.append(
                            np.asarray(result.observation["_cam_top_hd"]).copy()
                        )
                        history.append(current)
                        targets.append(raw_env.data.mocap_pos[raw_env.mocap_id, :2].copy())
                        blocks.append(raw_env.data.xpos[block_body, :2].copy())
                        end_effectors.append(
                            raw_env.data.xpos[end_effector_body, :2].copy()
                        )
                        if renderer is not None:
                            frames.append(render_frame(renderer, raw_env.data, label, seed, steps))
                        terminated = result.terminated
                        truncated = result.truncated
                        if terminated or truncated or steps >= options.max_steps:
                            break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    return Capture(
        label,
        seed,
        terminated,
        frames,
        top_frames,
        targets,
        blocks,
        goal,
        end_effectors,
        block_yaw,
        goal_yaw,
    )


def write_video(captures: tuple[Capture, Capture], output: Path) -> None:
    frames: list[np.ndarray] = []
    for capture in captures:
        title = np.full((720, 1280, 3), (8, 16, 28), dtype=np.uint8)
        cv2.putText(title, f"{capture.label} rollout", (160, 320), 0, 2.0, (255, 255, 255), 4)
        cv2.putText(title, f"seed {capture.seed} | 4x playback", (160, 385), 0, 1.0, (160, 220, 255), 2)
        frames.extend([title] * 20)
        frames.extend(capture.frames)
    iio.mimsave(output, frames, fps=40, macro_block_size=1)


def plot_trajectories(captures: tuple[Capture, Capture], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, capture in zip(axes, captures, strict=True):
        target = np.asarray(capture.targets)
        block = np.asarray(capture.blocks)
        outcome = "Success" if capture.success else "Failure"
        color = "#0f766e" if capture.success else "#dc2626"
        axis.plot(target[:, 0], target[:, 1], color=color, lw=2.6, label="policy mocap target")
        axis.plot(block[:, 0], block[:, 1], color="#7c3aed", lw=2.0, label="T-block")
        axis.scatter(*target[0], s=65, color=color, marker="o", label="target start")
        axis.scatter(*target[-1], s=90, color=color, marker="X", label="target end")
        axis.scatter(*block[0], s=65, color="#7c3aed", marker="o", label="block start")
        axis.scatter(*block[-1], s=90, color="#7c3aed", marker="X", label="block end")
        axis.scatter(*capture.goal, s=190, color="#f59e0b", marker="*", label="goal")
        axis.set(title=f"{outcome} | seed {capture.seed}", xlabel="world X (m)", ylabel="world Y (m)")
        axis.set_xlim(0.05, 0.45)
        axis.set_ylim(-0.22, 0.22)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_summary(metrics: dict[str, object], output: Path) -> None:
    rollouts = metrics["rollouts"]
    successes = sum(item["success"] is True for item in rollouts)
    failures = len(rollouts) - successes
    durations = [item["duration_s"] for item in rollouts]
    dxy = [item["dxy"] for item in rollouts]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(["success", "failure"], [successes, failures], color=["#0f766e", "#dc2626"])
    axes[0].set(title="100 seeded MuJoCo rollouts", ylabel="episodes", ylim=(0, 100))
    axes[0].text(0, successes + 3, f"{metrics['eval/success_rate']:.0%}", ha="center", weight="bold")
    axes[1].scatter(durations, dxy, c=["#0f766e" if item["success"] else "#dc2626" for item in rollouts], alpha=0.75)
    axes[1].set(title="Terminal position error", xlabel="duration (s)", ylabel="dxy (m)")
    axes[1].grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    assets = args.asset_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    policy, identity = load_policy(root, args.artifact, args.model)
    success = capture_rollout(policy, identity, args.success_seed, "SUCCESS")
    failure = capture_rollout(policy, identity, args.failure_seed, "FAILURE")
    if not success.success or failure.success:
        raise RuntimeError("selected seeds no longer match recorded success/failure outcomes")
    stem = "2026-08-20_dp_transformer_feedback"
    write_video((success, failure), assets / f"{stem}_3d_4x.mp4")
    plot_trajectories((success, failure), assets / f"{stem}_trajectories.png")
    metrics = json.loads((root / "inference/dp_transformer_eval/metrics.json").read_text())
    plot_summary(metrics, assets / f"{stem}_summary.png")
    print(f"published feedback assets under {assets}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
