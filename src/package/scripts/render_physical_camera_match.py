"""Render real, front-centered, and behind-robot camera geometry side by side."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from so101_pusht_benchmark.sim.physical_camera import (
    CALIBRATION_BLOCK_POSITION,
    calibration_overlay_xml,
    fit_camera_frame,
)


PROJECT_ROOT = Path("/home/intro/InternLab/02_InTro_Project")
ENVIRONMENT_XML = (
    PROJECT_ROOT
    / "05_references/external_repos/pushT-so100"
    / "chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"
)
WEBCAM_FRAME = PROJECT_ROOT / "04_experiments/camera_test/webcam_live_latest.jpg"
OUTPUT = PROJECT_ROOT / "04_experiments/camera_test/physical_environment_match_comparison.png"


def load_camera_model() -> mujoco.MjModel:
    """Compile frozen human_env with owned diagnostic cameras, without editing it."""
    xml = ENVIRONMENT_XML.read_text(encoding="utf-8").replace(
        "</worldbody>", f"{calibration_overlay_xml()}</worldbody>", 1
    )
    original_cwd = Path.cwd()
    os.chdir(ENVIRONMENT_XML.parent)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(original_cwd)


def recolor_physical_workspace(model: mujoco.MjModel) -> None:
    """Match real robot and target colors without mutating frozen XML files."""
    purple = np.array((0.35, 0.12, 0.48, 1.0), dtype=np.float32)
    red = np.array((0.82, 0.25, 0.27, 1.0), dtype=np.float32)
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] in (2, 3):
            model.geom_rgba[geom_id] = purple
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_sign")
    if target_id == -1:
        raise RuntimeError("frozen human_env does not contain T_sign")
    first = model.body_geomadr[target_id]
    count = model.body_geomnum[target_id]
    model.geom_rgba[first : first + count] = red


def place_block_like_live_workspace(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Move only the free T-block translation to the live camera's left-side pose."""
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
    if block_id == -1:
        raise RuntimeError("frozen human_env does not contain T_block")
    joint_id = model.body_jntadr[block_id]
    qpos_address = model.jnt_qposadr[joint_id]
    data.qpos[qpos_address : qpos_address + 3] = np.asarray(
        CALIBRATION_BLOCK_POSITION,
        dtype=np.float64,
    )


def render_camera(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Render one 4:3 frame using the requested fixed camera."""
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=name)
        return renderer.render()


def labelled(title: str, image: np.ndarray) -> np.ndarray:
    """Add a visible panel title without changing the camera image pixels."""
    panel = np.full((526, 640, 3), (19, 30, 46), dtype=np.uint8)
    panel[46:, :] = fit_camera_frame(image, width=640, height=480)
    cv2.putText(
        panel,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 245, 250),
        2,
    )
    return panel


def main() -> int:
    """Render one inspectable physical-camera comparison artifact."""
    model = load_camera_model()
    recolor_physical_workspace(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    place_block_like_live_workspace(model, data)
    mujoco.mj_forward(model, data)
    webcam = cv2.cvtColor(cv2.imread(str(WEBCAM_FRAME)), cv2.COLOR_BGR2RGB)
    comparison = np.concatenate(
        (
            labelled("REAL WEBCAM: front-center reference", webcam),
            labelled("SIM: calibrated front-center camera", render_camera(model, data, "physical_front")),
            labelled("SIM: behind-robot control (not selected)", render_camera(model, data, "behind_robot")),
        ),
        axis=1,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {OUTPUT}")
    print(f"Rendered {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
