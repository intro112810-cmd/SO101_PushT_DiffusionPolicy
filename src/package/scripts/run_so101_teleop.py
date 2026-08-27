"""Print or execute the calibrated SO-101 leader/follower teleoperation command."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

from so101_pusht_benchmark.hardware_live import device_holders, live_checks
from so101_pusht_benchmark.hardware_profile import load_hardware_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/hardware/so101_real_v1.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--max-relative-target-degrees", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def teleop_command(
    profile_path: Path, *, max_relative_target_degrees: float = 1.0, fps: int = 10
) -> list[str]:
    profile = load_hardware_profile(profile_path)
    camera = profile.camera
    camera_config = (
        "{ front: {type: opencv, "
        f"index_or_path: {camera.device}, width: {camera.width}, "
        f"height: {camera.height}, fps: {camera.fps}"
        "}}"
    )
    return [
        "/home/intro/miniforge3/envs/lerobot/bin/lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={profile.follower.port}",
        f"--robot.id={profile.follower.calibration_id}",
        f"--robot.max_relative_target={max_relative_target_degrees}",
        "--robot.disable_torque_on_disconnect=true",
        f"--robot.cameras={camera_config}",
        "--teleop.type=so101_leader",
        f"--teleop.port={profile.leader.port}",
        f"--teleop.id={profile.leader.calibration_id}",
        f"--fps={fps}",
        "--display_data=true",
    ]


def main() -> int:
    args = parse_args()
    profile_path = args.profile.resolve()
    profile = load_hardware_profile(profile_path)
    checks = live_checks(profile)
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"hardware preflight failed: {failed}")
    if not 0.0 < args.max_relative_target_degrees <= profile.max_relative_target_degrees:
        raise ValueError("max relative target must be positive and within the profile limit")
    if not 1 <= args.fps <= profile.camera.fps:
        raise ValueError("teleop fps must be within the camera/profile limit")
    command = teleop_command(
        profile_path,
        max_relative_target_degrees=args.max_relative_target_degrees,
        fps=args.fps,
    )
    print(shlex.join(command))
    if not args.execute:
        print("dry-run only: no robot connection or actuation was performed")
        return 0
    if profile.require_workspace_confirmation and not args.confirm_workspace_clear:
        raise RuntimeError("--confirm-workspace-clear is required for actuation")
    holders = device_holders(profile.camera.device)
    if holders:
        raise RuntimeError(
            "camera is already open; close preview/processes before actuation: "
            + ", ".join(str(process) for process in holders)
        )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
