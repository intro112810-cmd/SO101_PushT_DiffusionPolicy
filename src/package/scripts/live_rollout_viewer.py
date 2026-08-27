#!/usr/bin/env python3
"""Loop live rollouts of one trained bundle in the frozen MuJoCo scene.

Single-camera 96x96 policies loop 100000.. eagerly: each episode seeds 100000+i % 100,
terminates/truncates, then resets. Tops viewer uses DISPLAY if present, else EGL headless
frames stitched per observation history.

Usage:
  PUSHT_SINGLE_CAM=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=src:ART/cache/upstream/stanford:ART/cache/upstream/robomimic \
  /home/intro/miniforge3/envs/so100test/bin/python scripts/live_rollout_viewer.py \
    --artifact-root ART --artifact local-dp_transformer-seed0 --model dp_transformer --episodes 30 --seed 100000
"""
from __future__ import annotations
import argparse, itertools, os, time
from collections import deque
from pathlib import Path
import numpy as np, torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
from so101_pusht_benchmark.collection.viewer import LiveViewer, RealtimePacer
from so101_pusht_benchmark.integrations.paper_baselines.runner import PaperBaselineRunner, validate_native_runner_observation
from so101_pusht_benchmark.training.artifacts import ArtifactIndex, sha256_file
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity
from so101_pusht_benchmark.training.metadata import read_normalizer_metadata, read_trusted_config

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--viewer",
        action="store_true",
        help="show the live MuJoCo camera render at the native 10 Hz rollout rate",
    )
    p.add_argument(
        "--viewer-3d",
        action="store_true",
        help="show the adjustable native MuJoCo 3D viewer instead of camera frames",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="repeat viewer rollouts indefinitely without writing video artifacts",
    )
    return p.parse_args()

def main()->int:
    a=parse_args()
    root=a.artifact_root.resolve()
    index=ArtifactIndex(root / "artifact-index.json", root)
    record=index.record(a.artifact)
    identity=BundleIdentity.from_dict(record.get("identity"))
    if a.model != identity.model:
        raise RuntimeError(f"artifact {identity.model} != {a.model}")
    ckpt=index.verify(a.artifact, "checkpoint")
    cfg_path=index.verify(a.artifact, "config")
    norm_path=index.verify(a.artifact, "normalizer")
    bundle_path=index.verify(a.artifact, "bundle")
    ckpt_d=sha256_file(ckpt); cfg_d=sha256_file(cfg_path)
    config=read_trusted_config(cfg_path, identity.model)
    norm_state=read_normalizer_metadata(norm_path, identity, ckpt_d, cfg_d)
    policy=instantiate(OmegaConf.create(config["policy"]))
    expected=dict(policy.state_dict())
    dtypes={"torch.float32": torch.float32, "torch.float64": torch.float64}
    for k,(shape,dtype) in norm_state.items():
        expected[k]=torch.empty(shape, dtype=dtypes[dtype])
    state=load_bundle(bundle_path, expected, index=index, artifact_id=a.artifact, expectation=BundleExpectation(identity, ckpt_d))
    policy.load_state_dict(state, strict=True)
    policy.to(a.device); policy.eval()
    # infer history horizon from profile identity
    n_obs=identity.observation_steps
    n_act=identity.executed_actions
    if a.loop and not (a.viewer or a.viewer_3d):
        raise RuntimeError("loop playback requires --viewer or --viewer-3d")
    out_dir = None if a.loop else a.output_dir or (root / "inference_rollouts" / f"{a.model}_live")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as iio
    env=load_frozen_pusht(max_steps=a.steps)
    runner=PaperBaselineRunner(root / "tmp-live-rollout", evaluation_seeds=(), n_obs_steps=n_obs, n_action_steps=n_act, options={"max_steps": a.steps, "native_env_factory": "frozen"})
    camera_viewer = LiveViewer.open(
        enabled=a.viewer and not a.viewer_3d,
        title=f"MuJoCo policy rollout: {a.model}",
    )
    if a.viewer and not a.viewer_3d and not camera_viewer.enabled:
        raise RuntimeError("MuJoCo viewer requested but no graphical display is available")
    mujoco_viewer = None
    if a.viewer_3d:
        import mujoco.viewer

        raw_env = env._environment
        mujoco_viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
    pacer = RealtimePacer(time) if a.viewer or a.viewer_3d else None
    single=os.environ.get("PUSHT_SINGLE_CAM")=="1"
    ok=fail=0
    import cv2
    for i in itertools.count() if a.loop else range(a.episodes):
        seed=a.seed + (i % 100)
        obs_raw,_=env.reset(seed=seed)
        # HD frame for video (224 raw if patched, else 96)
        obs_hd = obs_raw.get("_cam_top_hd")
        obs_core={k: obs_raw[k] for k in obs_raw if not k.startswith("_")}
        obs=validate_native_runner_observation(obs_core if "_cam_top_hd" in obs_raw else obs_raw)
        hist=deque((obs for _ in range(n_obs)), maxlen=n_obs)
        policy.reset()
        frames=[] if not a.loop else None; terminated=truncated=False
        with torch.no_grad():
            for _ in range(a.steps):
                pred=policy.predict_action(runner.policy_observation(hist, policy))
                chunk=pred["action"].detach().cpu().numpy()
                for raw_action in chunk[0]:
                    action=np.asarray(raw_action, dtype=np.float32)
                    result=env.step(action)
                    nxt_raw=result.observation
                    hd_frame = nxt_raw.get("_cam_top_hd")
                    nxt_core={k: nxt_raw[k] for k in nxt_raw if not k.startswith("_")}
                    nxt=validate_native_runner_observation(nxt_core)
                    # HD video: upscale source (224) to 448 sharp; policy uses 96
                    if hd_frame is not None:
                        top_hd = cv2.resize(np.asarray(hd_frame), (448,448), interpolation=cv2.INTER_NEAREST)
                    else:
                        top_hd = cv2.resize(np.asarray(nxt["cam_top"]), (448,448), interpolation=cv2.INTER_NEAREST)
                    if "cam_side" in nxt_raw:
                        side_hd = cv2.resize(np.asarray(nxt_raw["cam_side"]), (448,448), interpolation=cv2.INTER_NEAREST)
                        display_frame = np.concatenate([top_hd, side_hd], axis=1)
                    else:
                        display_frame = top_hd
                    if frames is not None:
                        frames.append(display_frame)
                    if mujoco_viewer is not None:
                        if not mujoco_viewer.is_running():
                            return 0
                        mujoco_viewer.sync()
                    elif pacer is not None:
                        camera_viewer.show(display_frame)
                        pacer.wait()
                    if mujoco_viewer is not None and pacer is not None:
                        pacer.wait()
                    hist.append(nxt)
                    if result.terminated: terminated=True; break
                    if result.truncated: truncated=True; break
                if terminated or truncated: break
        ok += int(terminated); fail += int(not terminated)
        tag="ok" if terminated else "fail"
        if frames is not None:
            mp4=out_dir / f"rollout_{a.model}_seed{seed}_{tag}.mp4"
            iio.mimsave(mp4, frames, fps=10)
        print(f"[{i+1}{'/∞' if a.loop else f'/{a.episodes}'}] seed={seed} {tag} succ={ok}/{i+1}", flush=True)
        if a.loop:
            continue
    # summary json
    import json
    summary={"model":a.model,"artifact":a.artifact,"episodes":a.episodes,"seed_start":a.seed,"success":ok,"fail":fail,"success_rate": ok/a.episodes if a.episodes else 0}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding="utf-8")
    camera_viewer.close()
    if mujoco_viewer is not None:
        mujoco_viewer.close()
    print(f"done success_rate={summary['success_rate']:.1%} dir={out_dir}", flush=True)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
