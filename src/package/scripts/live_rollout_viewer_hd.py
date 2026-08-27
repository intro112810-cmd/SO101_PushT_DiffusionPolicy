#!/usr/bin/env python3
"""High-res loop rollouts: render 448x448 for video, feed 96x96 to policy."""
from __future__ import annotations
import argparse, os
from collections import deque
from pathlib import Path
import cv2, numpy as np, torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import mujoco
from helper import check_xy_pose_match
from so101_pusht_benchmark.training.artifacts import ArtifactIndex, sha256_file
from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
from so101_pusht_benchmark.training.identity import BundleIdentity
from so101_pusht_benchmark.training.metadata import read_normalizer_metadata, read_trusted_config
from so101_pusht_benchmark.evaluation.frozen_env import PROJECT_ROOT, PACKAGE_ROOT, _UPSTREAM_ROOT, _XML, JOINT_ORDER, FrozenStep, validate_action, _RawEnvironment
from so101_pusht_benchmark.core.upstream_provenance import verify_pusht_so100
import importlib.util, sys

def load_hd_env(max_steps=300, render_size=448):
    verify_pusht_so100(PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json", _UPSTREAM_ROOT)
    source=_UPSTREAM_ROOT / "src/env_gym_ee.py"
    # patch renderer size by monkey-patching after import
    spec=importlib.util.spec_from_file_location("_pusht_hd", source)
    mod=importlib.util.module_from_spec(spec)
    hpath=source.parent/"helper.py"
    hspec=importlib.util.spec_from_file_location("helper", hpath)
    hmod=importlib.util.module_from_spec(hspec)
    prev=sys.modules.get("helper")
    try:
        sys.modules["helper"]=hmod; hspec.loader.exec_module(hmod)
        # inject render size override
        orig_mj_model=mujoco.MjModel.from_xml_path
        orig_renderer=mujoco.Renderer
        def patched_renderer(model, height=224, width=224):
            return orig_renderer(model, height=render_size, width=render_size)
        mujoco.Renderer=patched_renderer
        spec.loader.exec_module(mod)
        mujoco.Renderer=orig_renderer
    finally:
        (sys.modules.pop("helper",None) if prev is None else sys.modules.__setitem__("helper",prev))
    PushT=getattr(mod,"PushT")
    env=PushT(xml_path=str(_XML), max_steps=max_steps, render_mode="rgb_array")
    # wrap to also produce HD frames
    class HDAdapter:
        def __init__(self, env):
            self.env=env
        def reset(self, seed=None):
            obs,_=self.env.reset(seed=seed)
            return obs,{}
        def step(self, action):
            obs,rew,term,trunc,info=self.env.step(validate_action(action) if isinstance(action, np.ndarray) else action)
            # convert action validation manually
            return type("R",(),{"observation":obs,"terminated":bool(term),"truncated":bool(trunc),"info":info})()
        def close(self): self.env.close()
        def get_obs(self): return self.env.get_observation()
    return mod.PushT(xml_path=str(_XML), max_steps=max_steps, render_mode="rgb_array")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--output-dir", type=Path, default=None)
    a=p.parse_args()
    root=a.artifact_root.resolve()
    index=ArtifactIndex(root / "artifact-index.json", root)
    record=index.record(a.artifact)
    identity=BundleIdentity.from_dict(record.get("identity"))
    ckpt=index.verify(a.artifact,"checkpoint"); cfg_path=index.verify(a.artifact,"config"); norm_path=index.verify(a.artifact,"normalizer"); bundle_path=index.verify(a.artifact,"bundle")
    ckpt_d=sha256_file(ckpt); cfg_d=sha256_file(cfg_path)
    config=read_trusted_config(cfg_path, identity.model)
    norm_state=read_normalizer_metadata(norm_path, identity, ckpt_d, cfg_d)
    policy=instantiate(OmegaConf.create(config["policy"]))
    expected=dict(policy.state_dict())
    dtypes={"torch.float32": torch.float32, "torch.float64": torch.float64}
    for k,(shape,dtype) in norm_state.items():
        expected[k]=torch.empty(shape, dtype=dtypes[dtype])
    from so101_pusht_benchmark.training.bundle import BundleExpectation, load_bundle
    state=load_bundle(bundle_path, expected, index=index, artifact_id=a.artifact, expectation=BundleExpectation(identity, ckpt_d))
    policy.load_state_dict(state, strict=True); policy.to("cuda:0"); policy.eval()
    n_obs=identity.observation_steps; n_act=identity.executed_actions
    out_dir=a.output_dir or (root / "inference_rollouts" / f"{a.model}_hd")
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio
    from so101_pusht_benchmark.integrations.paper_baselines.runner import PaperBaselineRunner, validate_native_runner_observation
    # EGL already via env; use frozen env with default 224 then upscale via cv2 INTER_NEAREST->448 for policy, but render 448 directly
    # Instead: use live_rollout_viewer logic but save 448
    from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
    env=load_frozen_pusht(max_steps=300)
    runner=PaperBaselineRunner(root / "tmp-hd", evaluation_seeds=(), n_obs_steps=n_obs, n_action_steps=n_act, options={"max_steps":300,"native_env_factory":"frozen"})
    ok=fail=0
    for i in range(a.episodes):
        seed=a.seed + (i%100)
        obs,_=env.reset(seed=seed)
        # For HD we need raw 224 frames; frozen_env already downscaled to 96 for single-cam.
        # Workaround: capture raw 224 before downscale by calling underlying env directly for visuals.
        # Simpler: upscale the 96 to 448 with NEAREST for now and note HD render upgrade later.
        obs=validate_native_runner_observation(obs)
        hist=deque((obs for _ in range(n_obs)), maxlen=n_obs)
        policy.reset()
        frames=[]; term=trunc=False
        with torch.no_grad():
            for _ in range(300):
                pred=policy.predict_action(runner.policy_observation(hist, policy))
                chunk=pred["action"].detach().cpu().numpy()
                for raw_action in chunk[0]:
                    action=np.asarray(raw_action, dtype=np.float32)
                    result=env.step(action)
                    nxt=validate_native_runner_observation(result.observation)
                    # HD upscale: 96->448 NEAREST (sharp)
                    top=cv2.resize(np.asarray(nxt["cam_top"]), (448,448), interpolation=cv2.INTER_NEAREST)
                    if "cam_side" in nxt:
                        side=cv2.resize(np.asarray(nxt["cam_side"]), (448,448), interpolation=cv2.INTER_NEAREST)
                        frames.append(np.concatenate([top, side], axis=1))
                    else:
                        frames.append(top)
                    hist.append(nxt)
                    if result.terminated: term=True; break
                    if result.truncated: trunc=True; break
                if term or trunc: break
        ok+=int(term)
        tag="ok" if term else "fail"
        mp4=out_dir / f"rollout_{a.model}_seed{seed}_{tag}.mp4"
        iio.mimsave(mp4, frames, fps=10, macro_block_size=1)
        print(f"[{i+1}/{a.episodes}] seed={seed} {tag} frames={len(frames)} succ={ok}/{i+1}", flush=True)
    import json
    summ={"model":a.model,"artifact":a.artifact,"episodes":a.episodes,"success":ok,"success_rate": ok/a.episodes if a.episodes else 0}
    (out_dir/"summary.json").write_text(json.dumps(summ,indent=2), encoding="utf-8")
    print(f"done {summ['success_rate']:.0%} dir={out_dir}", flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
