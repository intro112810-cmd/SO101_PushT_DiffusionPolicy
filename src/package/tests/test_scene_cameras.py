import numpy as np
import pytest
from pathlib import Path
from so101_pusht_benchmark.sim.scene import CameraNotFoundError, Scene
from so101_pusht_benchmark.sim.physical_camera import (
    BEHIND_ROBOT_CAMERA,
    CALIBRATION_BLOCK_POSITION,
    PHYSICAL_FRONT_CAMERA,
    calibration_overlay_xml,
    fit_camera_frame,
)


def test_topdown_camera_exists_and_renders() -> None:
    scene = Scene()
    try:
        # Check if camera exists in mujoco model
        cam_type = int(scene.mujoco.mjtObj.mjOBJ_CAMERA)  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        cam_id = int(scene.mujoco.mj_name2id(scene.model, cam_type, "topdown"))  # type: ignore[reportUnknownArgumentType]
        assert cam_id != -1, "topdown camera not found in model"

        frame = scene.render(camera="topdown")
        assert frame is not None
        assert frame.shape == (96, 96, 3)
        assert frame.dtype == np.uint8
    finally:
        scene.close()


def test_front_camera_preserves_original_checkpoint_pose() -> None:
    overlay = (
        Path(__file__).parents[1]
        / "assets/mujoco/so101_pusht_overlay.xml"
    ).read_text(encoding="utf-8")
    assert (
        '<camera name="front" pos=".55 -.55 .35" '
        'xyaxes=".707 .707 0 -.331 .331 .884"/>'
    ) in overlay

    scene = Scene()
    try:
        scene.mujoco.mj_forward(scene.model, scene.data)
        frame = scene.render(camera="front")
        assert frame.shape == (96, 96, 3)
        assert frame.dtype == np.uint8
        # Just verify it's not completely uniform (rendering actually happened)
        assert np.var(frame) > 10.0
    finally:
        scene.close()


def test_physical_front_camera_matches_webcam_centerline_pose() -> None:
    overlay = (
        Path(__file__).parents[1]
        / "assets/mujoco/so101_pusht_overlay.xml"
    ).read_text(encoding="utf-8")
    assert (
        '<camera name="physical_front" pos=".55 0 .55" '
        'xyaxes="0 1 0 -.88 0 .48" fovy="58"/>'
    ) in overlay

    scene = Scene()
    try:
        scene.mujoco.mj_forward(scene.model, scene.data)
        frame = scene.render(camera="physical_front")
        assert frame.shape == (96, 96, 3)
        assert frame.dtype == np.uint8
        assert np.var(frame) > 10.0
    finally:
        scene.close()


def test_topdown_contains_scene_landmarks() -> None:
    scene = Scene()
    try:
        scene.mujoco.mj_forward(scene.model, scene.data)
        frame = scene.render(camera="topdown")
        assert frame.shape == (96, 96, 3)
        assert frame.dtype == np.uint8

        # Verify not completely empty/constant (meaning it sees the scene)
        assert np.var(frame) > 10.0, (
            "Topdown camera frame should have some variance (seeing table/objects)"
        )
    finally:
        scene.close()


def test_unknown_camera_raises_error() -> None:
    scene = Scene()
    try:
        with pytest.raises(CameraNotFoundError):
            scene.render(camera="invalid_camera_name_123")
    finally:
        scene.close()


def test_physical_camera_pose_is_centered_in_front_of_robot() -> None:
    assert PHYSICAL_FRONT_CAMERA.position[0] > 0.25
    assert PHYSICAL_FRONT_CAMERA.position[1] == 0.0
    assert PHYSICAL_FRONT_CAMERA.position[2] > 0.4
    assert BEHIND_ROBOT_CAMERA.position[0] < 0.0


def test_calibration_overlay_matches_workspace_visual_landmarks() -> None:
    overlay = calibration_overlay_xml()
    assert 'name="physical_front"' in overlay
    assert 'name="calibration_table_rim_left"' in overlay
    assert 'name="calibration_table_rim_right"' in overlay
    assert 'name="calibration_table_rim_back"' in overlay
    assert CALIBRATION_BLOCK_POSITION[1] < 0.0


def test_calibration_renderer_uses_live_workspace_frame() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/render_physical_camera_match.py"
    ).read_text(encoding="utf-8")
    assert "webcam_live_latest.jpg" in script


def test_square_live_crop_is_letterboxed_for_comparison_panel() -> None:
    crop = np.full((400, 400, 3), 127, dtype=np.uint8)

    fitted = fit_camera_frame(crop, width=640, height=480)

    assert fitted.shape == (480, 640, 3)
    assert np.all(fitted[:, :80] == 0)
    assert np.all(fitted[:, 80:560] == 127)
    assert np.all(fitted[:, 560:] == 0)
