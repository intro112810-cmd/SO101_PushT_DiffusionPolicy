"""Post-capture non-actuating pipeline command contract."""

from pathlib import Path

from so101_pusht_benchmark.sim_to_real.post_capture_pipeline import (
    PostCapturePaths,
    build_commands,
)


def test_build_commands_orders_non_actuating_gates() -> None:
    paths = PostCapturePaths.fixture(Path("scratch/root"))
    commands = build_commands(paths, python="python")
    assert [Path(command[1]).name for command in commands] == [
        "build_camera_corpus_from_video.py",
        "issue_camera_corpus_authority_offline.py",
        "prepare_joint_corpus_authority.py",
        "audit_camera_registration.py",
        "audit_joint_equivalence_read_only.py",
        "bind_hardware_profile.py",
        "run_real_shadow_inference.py",
    ]
    assert "--hardware-profile" in commands[-1]
    assert commands[-1][-2:] == ["--output", str(paths.shadow_output)]


def test_every_output_is_fresh_and_under_run_root() -> None:
    paths = PostCapturePaths.fixture(Path("scratch/root"))
    assert set(paths.outputs()) == {
        paths.camera_output_dir,
        paths.camera_authority_dir,
        paths.joint_authority_dir,
        paths.camera_receipt,
        paths.joint_receipt,
        paths.bound_profile,
        paths.shadow_output,
    }
