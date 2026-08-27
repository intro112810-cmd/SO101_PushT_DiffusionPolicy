"""Compose read-only physical inference with an isolated MuJoCo preview."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

os.environ.setdefault("PUSHT_SINGLE_CAM", "1")
os.environ.setdefault("PUSHT_LOCAL_BUDGET", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from so101_pusht_benchmark.sim_to_real.contracts import (
    ContractError,
    build_dry_run_contract,
    validate_follower_receipt,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    locate_receipt_path,
    ReceiptPathIdentity,
    validate_receipt_identity,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.preview import (
    render_prediction_preview,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("dp_cnn",))
    parser.add_argument("--artifact")
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--follower-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy-seed", type=int, default=100018)
    parser.add_argument("--preview-seed", type=int, default=100018)
    parser.add_argument("--preview-mp4", type=Path)
    parser.add_argument("--preview-png", type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON mapping")
    return cast("dict[str, object]", raw)


def _preview_locations(
    args: argparse.Namespace,
    output_location: ReceiptPathIdentity,
) -> tuple[ReceiptPathIdentity, ReceiptPathIdentity] | None:
    if args.preview_png is None or args.preview_mp4 is None:
        return None
    png = validate_receipt_identity(
        locate_receipt_path(args.preview_png), production=output_location.canonical
    )
    mp4 = validate_receipt_identity(
        locate_receipt_path(args.preview_mp4), production=output_location.canonical
    )
    if png.resolved == mp4.resolved:
        raise ValueError("preview PNG and MP4 must use distinct paths")
    if any(item.resolved.exists() or item.resolved.is_symlink() for item in (png, mp4)):
        raise ValueError("preview output already exists")
    return png, mp4


def _remove_preview_pair(
    locations: tuple[ReceiptPathIdentity, ReceiptPathIdentity] | None,
) -> None:
    if locations is None:
        return
    for location in locations:
        if location.resolved.is_file() and not location.resolved.is_symlink():
            location.resolved.unlink()


def run_dry_run(args: argparse.Namespace) -> dict[str, object]:
    follower_path = args.follower_state.resolve()
    follower = _json(follower_path)
    validate_follower_receipt(follower)
    if (args.preview_mp4 is None) != (args.preview_png is None):
        raise ValueError("--preview-mp4 and --preview-png must be provided together")
    output_location = locate_receipt_path(args.output_dir)
    preview_locations = _preview_locations(args, output_location)
    output_dir = output_location.lexical
    shadow_path = output_dir / "shadow_inference.json"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_real_shadow_inference.py"),
        "--artifact-root",
        str(args.artifact_root.resolve()),
        "--model",
        args.model,
        "--frame",
        str(args.frame.resolve()),
        "--samples",
        str(args.samples.resolve()),
        "--lineage",
        str(args.lineage.resolve()),
        "--joint",
        str(args.joint.resolve()),
        "--camera",
        str(args.camera.resolve()),
        "--policy-seed",
        str(args.policy_seed),
        "--output",
        str(shadow_path),
    ]
    if args.artifact:
        command.extend(("--artifact", args.artifact))
    subprocess.run(command, check=True, env=os.environ.copy())
    shadow = _json(output_location.resolved / "shadow_inference.json")
    receipt = build_dry_run_contract(follower, shadow)
    production = shadow.get("evidence_scope") == "production"
    receipt_location = locate_receipt_path(output_dir / "dry_run_receipt.json")
    validate_receipt_identity(receipt_location, production=production)
    if preview_locations is not None:
        for location in preview_locations:
            validate_receipt_identity(location, production=production)
    try:
        if preview_locations is not None:
            png_location, mp4_location = preview_locations
            actions = cast("list[list[float]]", receipt["predicted_actions"])
            receipt["preview"] = render_prediction_preview(
                actions,
                png_path=png_location.resolved,
                mp4_path=mp4_location.resolved,
                seed=args.preview_seed,
                evidence_scope="production" if production else "test_fixture_only",
            )
        receipt_location.resolved.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        _remove_preview_pair(preview_locations)
        raise
    return receipt


def main() -> int:
    args = parse_args()
    try:
        receipt = run_dry_run(args)
    except (OSError, ValueError, TypeError, ContractError, RolloutViolation) as exc:
        print(f"R_MISSING: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"R_MISSING: shadow inference failed with exit {exc.returncode}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
