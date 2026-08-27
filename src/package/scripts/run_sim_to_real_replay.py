#!/usr/bin/env python3
"""Assemble an authentic two-step history and replay the frozen policy.

Production lineage always attempts the real frozen policy load and fails
closed when the installed runtime is missing ``CropRandomizer``. The fixture
lineage selects a fixture-only deterministic adapter after the observation
contract (uint8[96,96,3] frames and float32[5] states) is already validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import cast

from so101_pusht_benchmark.sim_to_real.receipt_routing import ReceiptRoutingError
from so101_pusht_benchmark.sim_to_real.replay_history import (
    HistoryEvidence,
    build_history,
    build_receipt,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
    write_receipt,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

REPLAY_LINEAGE_AUTHORITY_DIGEST = "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"
REPLAY_SOURCE_FRAME = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/sim_to_real/physical_frame.png"
)


class ReplayCLIError(RuntimeError):
    """CLI boundary failure carrying a closed rejection code."""

    def __init__(self, code: RolloutCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _json_mapping(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayCLIError(RolloutCode.R_MISSING, f"cannot read {path}") from exc
    if not isinstance(raw, dict):
        raise ReplayCLIError(RolloutCode.R_MISSING, f"{path} must be a JSON mapping")
    return cast("dict[str, object]", raw)


def main() -> int:
    args = parse_args()
    try:
        lineage_document = _json_mapping(args.lineage)
        samples_document = _json_mapping(args.samples)
        raw_samples = samples_document.get("samples")
        if not isinstance(raw_samples, list):
            raise ReplayCLIError(
                RolloutCode.HISTORY_INCOMPLETE, "samples receipt lacks a sample list"
            )
        typed_samples: list[dict[str, object]] = []
        for item in cast("list[object]", raw_samples):
            if not isinstance(item, dict):
                raise ReplayCLIError(
                    RolloutCode.HISTORY_INCOMPLETE, "samples receipt lacks sample mappings"
                )
            typed_samples.append(cast("dict[str, object]", item))
        lineage = validate_lineage_receipt(
            lineage_document,
            expected_digest=REPLAY_LINEAGE_AUTHORITY_DIGEST,
        )
        joint_document = _json_mapping(args.joint)
        camera_document = _json_mapping(args.camera)
        joint_digest = validate_joint_receipt(joint_document)
        camera_digest = validate_camera_receipt(camera_document)
        evidence = HistoryEvidence(
            samples=tuple(typed_samples),
            joint_document=joint_document,
            camera_document=camera_document,
            lineage_document=lineage_document,
            lineage_authority_digest=cast(str, lineage["authority_digest"]),
            source_frame_path=REPLAY_SOURCE_FRAME,
        )
        history = build_history(evidence)
        receipt = build_receipt(
            history,
            lineage=lineage,
            joint_digest=joint_digest,
            camera_digest=camera_digest,
            policy_seed=0,
        )
        write_receipt(receipt, args.output)
    except (ReceiptRoutingError, RolloutViolation, ReplayCLIError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_document(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
