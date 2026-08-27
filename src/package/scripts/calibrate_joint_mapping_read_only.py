"""Generate fail-closed joint-mapping evidence from existing read-only inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import cast

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from PIL import Image, ImageDraw

from so101_pusht_benchmark.sim.scene import Scene
from so101_pusht_benchmark.sim_to_real.joint_mapping import (
    JOINT_ORDER,
    JointMappingReceipt,
    build_joint_mapping_receipt,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import prepare_receipt_directory


DIAGNOSTIC_LABEL = "DIAGNOSTIC INVALID - NO COMMAND"
RECEIPT_NAME = "joint_mapping_receipt.json"
PNG_NAME = "physical_front_joint_mapping_diagnostic.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-file", required=True, type=Path)
    parser.add_argument("--follower-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="label synthetic inputs as fixture-only and forbid canonical evidence routing",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON mapping")
    return cast("dict[str, object]", raw)


def _calibration(path: Path) -> dict[str, dict[str, int]]:
    raw = _json(path)
    calibration: dict[str, dict[str, int]] = {}
    for joint in JOINT_ORDER:
        record = raw.get(joint)
        if not isinstance(record, dict):
            raise TypeError(f"calibration for {joint} must be a mapping")
        typed_record = cast("dict[str, object]", record)
        range_min = typed_record.get("range_min")
        range_max = typed_record.get("range_max")
        if (
            isinstance(range_min, bool)
            or not isinstance(range_min, int)
            or isinstance(range_max, bool)
            or not isinstance(range_max, int)
        ):
            raise TypeError(f"calibration for {joint} requires integer range endpoints")
        calibration[joint] = {"range_min": range_min, "range_max": range_max}
    return calibration


def _mujoco_ranges(scene: Scene) -> dict[str, tuple[float, float]]:
    joint_type = int(scene.mujoco.mjtObj.mjOBJ_JOINT)
    ranges: dict[str, tuple[float, float]] = {}
    for joint in JOINT_ORDER:
        joint_id = int(scene.mujoco.mj_name2id(scene.model, joint_type, joint))
        if joint_id == -1:
            raise RuntimeError(f"MuJoCo model is missing joint {joint}")
        ranges[joint] = (
            float(scene.model.jnt_range[joint_id, 0]),
            float(scene.model.jnt_range[joint_id, 1]),
        )
    return ranges


def _place_unclipped_diagnostic_pose(
    scene: Scene,
    receipt: JointMappingReceipt,
) -> None:
    joint_type = int(scene.mujoco.mjtObj.mjOBJ_JOINT)
    for joint in JOINT_ORDER:
        joint_id = int(scene.mujoco.mj_name2id(scene.model, joint_type, joint))
        qpos_address = int(scene.model.jnt_qposadr[joint_id])
        scene.data.qpos[qpos_address] = receipt["joints"][joint]["mapped_q_radians"]
    scene.mujoco.mj_forward(scene.model, scene.data)


def _annotate_diagnostic(image: Image.Image, receipt: JointMappingReceipt) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 104), fill=(10, 10, 10))
    draw.text((18, 14), DIAGNOSTIC_LABEL, fill=(255, 45, 45), stroke_width=1)
    elbow = receipt["joints"]["elbow_flex"]
    degree_min, degree_max = elbow["physical_degree_range"]
    q_min, q_max = elbow["mujoco_range_radians"]
    draw.text(
        (18, 42),
        (
            f"elbow physical {elbow['physical_degree']:.6f} deg "
            f"outside [{degree_min:.6f}, {degree_max:.6f}]"
        ),
        fill=(255, 220, 120),
    )
    draw.text(
        (18, 68),
        (
            f"unclipped affine q {elbow['mapped_q_radians']:.8f} rad "
            f"outside [{q_min:.5f}, {q_max:.5f}]"
        ),
        fill=(255, 220, 120),
    )
    return canvas


def main() -> int:
    args = parse_args()
    calibration_path = args.calibration_file.resolve()
    follower_path = args.follower_state.resolve()
    output_dir = prepare_receipt_directory(
        args.output_dir,
        production=not args.fixture_only,
    )
    png_path = output_dir / PNG_NAME
    receipt_path = output_dir / RECEIPT_NAME

    scene = Scene()
    try:
        receipt = build_joint_mapping_receipt(
            calibration=_calibration(calibration_path),
            follower_receipt=_json(follower_path),
            mujoco_ranges=_mujoco_ranges(scene),
        )
        _place_unclipped_diagnostic_pose(scene, receipt)
        frame = scene.render(camera="physical_front", size=480)
    finally:
        scene.close()

    annotated = _annotate_diagnostic(Image.fromarray(frame), receipt)
    annotated.save(png_path)
    persisted: dict[str, object] = dict(receipt)
    persisted.update(
        {
            "calibration_file": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "follower_state_receipt": str(follower_path),
            "follower_state_sha256": _sha256(follower_path),
            "mujoco_model": "pinned so101_new_calib.xml",
            "camera": "physical_front",
            "diagnostic_png": str(png_path),
            "rendered_pose": "unclipped affine values in isolated MuJoCo only",
            "evidence_scope": (
                "test_fixture_only" if args.fixture_only else "production_physical_diagnostic"
            ),
        }
    )
    receipt_path.write_text(json.dumps(persisted, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(persisted, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
