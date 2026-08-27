from __future__ import annotations
import numpy as np
from so101_pusht_benchmark.sim.env import PushTEnv


def test_abort_neutralizes_target_ctrl_and_never_steps() -> None:
    env = PushTEnv()
    env.reset(0)
    env.scene.data.mocap_pos[0] = (0.31, 0.1, 0.05)
    before = float(env.scene.data.time)
    env.abort_collection("test")
    assert float(env.scene.data.time) == before
    assert np.array_equal(env.scene.data.ctrl, env.scene.data.qpos[: env.scene.data.ctrl.size])
    assert env.filter.last.tolist() == env.scene.data.mocap_pos[0].tolist()
    env.close()
