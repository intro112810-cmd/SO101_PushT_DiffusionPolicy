"""Typed offline CLI for deterministic intrinsic-frame extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .camera_registration_vision import detect_checkerboard
from .intrinsic_extraction import ExtractionError
from .intrinsic_extraction_io import calibrate_and_evaluate, decode_video
from .intrinsic_extraction_pipeline import run_extraction
from .intrinsic_extraction_types import ExtractionDependencies, ExtractionRequest


class _Arguments(argparse.Namespace):
    source_video: Path
    expected_sha256: str
    output_directory: Path | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially evaluate every video frame with the current 3x ChArUco detector, "
            "deduplicate poses, compare deterministic fit sizes, and publish offline evidence."
        )
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Fresh output child; default is <source-stem>-extracted next to the source.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv, namespace=_Arguments())
    output = args.output_directory or args.source_video.with_name(
        f"{args.source_video.stem}-extracted"
    )
    try:
        receipt = run_extraction(
            ExtractionRequest(args.source_video, args.expected_sha256, output),
            ExtractionDependencies(decode_video, detect_checkerboard, calibrate_and_evaluate),
        )
    except (ExtractionError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_directory": str(output.absolute()),
                "source_sha256": receipt.source_sha256,
                "total_decoded": receipt.total_decoded,
                "fit_frame_count": receipt.fit_frame_count,
                "heldout_frame_count": receipt.heldout_frame_count,
                "minimum_pool_distance": receipt.minimum_pool_distance,
                "calibration_rms_px": receipt.evaluation.fit.rms_reprojection_error_px,
                "heldout_rms_px": receipt.evaluation.heldout.rms_error_px,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
