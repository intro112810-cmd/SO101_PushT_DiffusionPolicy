"""Read calibrated SO-101 follower positions without configuring or commanding motors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from lerobot.robots.so_follower.config_so_follower import (
    SOFollowerRobotConfig,
)
from lerobot.robots.so_follower.so_follower import SOFollower

from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    prepare_receipt_directory,
    validate_receipt_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--calibration-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    calibration_file = args.calibration_file.resolve()
    try:
        output = validate_receipt_path(args.output, production=True)
        prepare_receipt_directory(output.parent, production=True)
    except ReceiptRoutingError as exc:
        print(f"R_MISSING: {exc}", file=sys.stderr)
        return 2
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.calibration_id,
            cameras={},
            use_degrees=True,
        )
    )
    robot.bus.connect()
    try:
        positions = robot.bus.sync_read(
            "Present_Position",
            normalize=True,
        )
        raw_encoder = robot.bus.sync_read(
            "Present_Position",
            normalize=False,
        )
    finally:
        robot.bus.disconnect(disable_torque=False)
    receipt = {
        "schema": 1,
        "mode": "read_only_follower_state",
        "evidence_scope": "production_physical_diagnostic",
        "port": args.port,
        "calibration_id": args.calibration_id,
        "calibration_file": str(calibration_file),
        "calibration_sha256": sha256_file(calibration_file),
        "positions_degrees": positions,
        "raw_encoder": raw_encoder,
        "motor_writes_performed": False,
        "actuation_performed": False,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
