from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
from numpy.typing import NDArray

from so101_pusht_benchmark.sim.env import PushTEnv
from so101_pusht_benchmark.sim.scene import Scene, SceneError


def test_scene_loads_and_missing_source_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = Scene()
    assert scene.model.opt.timestep == pytest.approx(0.002)
    scene.close()
    monkeypatch.setattr("so101_pusht_benchmark.sim.scene.UPSTREAM", Path(__file__))
    with pytest.raises(SceneError):
        Scene()


def test_reset_and_front_rgb_are_deterministic_and_independent() -> None:
    first, second = PushTEnv(), PushTEnv()
    one, _ = first.reset(0)
    two, _ = second.reset(0)
    assert one["observation.images.front"].dtype == np.uint8
    assert one["observation.images.front"].shape == (96, 96, 3)
    assert np.array_equal(one["observation.images.front"], two["observation.images.front"])
    state = one["observation.state"]
    assert state.shape == (15,)
    assert state.dtype == np.float32
    assert np.array_equal(state[:6], first.scene.data.qpos[:6].astype(np.float32))
    assert np.array_equal(state[6:12], first.scene.data.qvel[:6].astype(np.float32))
    assert np.array_equal(state[12:], first.scene.data.xpos[first.pusher].astype(np.float32))
    frame = first.observe()["observation.images.front"]
    frame.fill(0)
    assert bool(first.observe()["observation.images.front"].any())
    first.close()
    second.close()


def test_happy_real_contact_moves_and_rotates_t_without_fault() -> None:
    env = PushTEnv(development=True)
    env.reset(0)
    start = env.scene.data.qpos[6:9].copy()
    pusher: NDArray[np.float64] = np.asarray(env.scene.data.mocap_pos[0], dtype=np.float64)
    qpos_xy: NDArray[np.float64] = np.asarray(env.scene.data.qpos[6:8], dtype=np.float64)
    direction: NDArray[np.float64] = np.asarray((0.0, 0.0), dtype=np.float64)
    direction[0] = float(qpos_xy[0]) - float(pusher[0])
    direction[1] = float(qpos_xy[1]) - float(pusher[1])
    norm = float(np.linalg.norm(direction))
    direction[0] = float(direction[0]) / norm
    direction[1] = float(direction[1]) / norm
    terminal = False
    for distance in (0.1, 0.12):
        target = pusher.copy()
        target[:2] += direction * distance
        target[2] = 0.045
        for _ in range(8):
            result = env.step(np.asarray(target, dtype=np.float32))
            terminal = result.terminated
            if terminal:
                break
    moved = np.linalg.norm(env.scene.data.qpos[6:8] - start[:2])
    yaw = env.scene.data.qpos[8] - start[2]
    assert not terminal
    assert moved > 0.01
    assert abs(yaw) > 0.001
    assert env.frame == 16
    assert env.scene.data.time == pytest.approx(1.6)
    env.close()


def test_bad_action_does_not_mutate_live_physics_and_latches() -> None:
    env = PushTEnv()
    env.reset(1)
    before = env.scene.data.qpos.copy()
    result = env.step(np.asarray([np.nan, 0, 0.05], dtype=np.float32))
    assert result.info["fault"] == "invalid_action"
    assert np.array_equal(before, env.scene.data.qpos)
    assert env.step(np.zeros(3, dtype=np.float32)).info["fault"] == "terminal"
    env.close()


def test_nonfinite_physics_latches() -> None:
    env = PushTEnv()
    env.reset(2)
    env.scene.data.qpos[0] = np.nan
    result = env.step(np.asarray([0.28, 0, 0.05], dtype=np.float32))
    assert result.terminated
    assert result.info["fault"] in {"nonfinite_physics", "invalid_ik"}
    env.close()


def test_calibration_rejects_altered_unsafe_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from so101_pusht_benchmark.sim import calibration

    altered = tmp_path / "unsafe.yaml"
    altered.write_text(
        calibration.CALIBRATION.read_text().replace(
            "joint_command_delta_rad: 1.2", "joint_command_delta_rad: 0.01"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(calibration, "CALIBRATION", altered)
    with pytest.raises(RuntimeError, match="outside measured safe set"):
        calibration.sweep(0)
