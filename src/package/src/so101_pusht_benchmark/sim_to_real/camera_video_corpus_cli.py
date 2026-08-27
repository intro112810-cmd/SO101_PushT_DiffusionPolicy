"""CLI for deterministic camera corpus assembly from recorded videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .camera_registration_capture_cli import preflight_registration_authority
from .camera_video_corpus import RecordedTableClip, VideoCorpusRequest, build_recorded_camera_corpus
from .rollout_codes import RolloutViolation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--acquisition-authority", required=True, type=Path)
    parser.add_argument("--authority-signature", required=True, type=Path)
    parser.add_argument("--trust-anchor", required=True, type=Path)
    parser.add_argument("--intrinsic-evidence", required=True, type=Path)
    parser.add_argument("--table-fit-a", required=True, type=Path)
    parser.add_argument("--table-fit-b", required=True, type=Path)
    parser.add_argument("--table-fit-c", required=True, type=Path)
    parser.add_argument("--checkpoint-held-a", required=True, type=Path)
    parser.add_argument("--checkpoint-held-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--measured-square-mm", required=True, type=float)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        authority = preflight_registration_authority(args)
        clips = tuple(
            RecordedTableClip(identifier, getattr(args, field))
            for identifier, field in (
                ("table-fit-a", "table_fit_a"),
                ("table-fit-b", "table_fit_b"),
                ("table-fit-c", "table_fit_c"),
                ("checkpoint-held-a", "checkpoint_held_a"),
                ("checkpoint-held-b", "checkpoint_held_b"),
            )
        )
        summary = build_recorded_camera_corpus(
            VideoCorpusRequest(
                args.intrinsic_evidence,
                clips,
                args.output_dir,
                args.measured_square_mm,
            ),
            authority,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RolloutViolation) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run()
