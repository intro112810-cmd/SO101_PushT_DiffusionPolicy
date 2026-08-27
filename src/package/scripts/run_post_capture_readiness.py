#!/usr/bin/env python3
"""Run or print the complete post-capture non-actuating readiness chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

from so101_pusht_benchmark.sim_to_real.post_capture_pipeline import (
    PostCapturePaths,
    build_commands,
    execute_pipeline,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-root",
        "profile-template",
        "acquisition-authority",
        "acquisition-signature",
        "trust-anchor",
        "intrinsic-evidence",
        "table-fit-a",
        "table-fit-b",
        "table-fit-c",
        "checkpoint-held-a",
        "checkpoint-held-b",
        "camera-policy",
        "private-key",
        "joint-corpus",
        "joint-policy",
        "lineage",
        "artifact-root",
        "frame",
        "samples",
    ):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--measured-square-mm", type=float, default=25.0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    paths = PostCapturePaths(
        a.run_root,
        a.profile_template,
        a.acquisition_authority,
        a.acquisition_signature,
        a.trust_anchor,
        a.intrinsic_evidence,
        a.table_fit_a,
        a.table_fit_b,
        a.table_fit_c,
        a.checkpoint_held_a,
        a.checkpoint_held_b,
        a.camera_policy,
        a.private_key,
        a.joint_corpus,
        a.joint_policy,
        a.lineage,
        a.artifact_root,
        a.frame,
        a.samples,
        a.measured_square_mm,
    )
    try:
        if a.execute:
            execute_pipeline(paths, python=a.python)
        else:
            print(
                json.dumps(
                    {
                        "actuation_performed": False,
                        "commands": build_commands(paths, python=a.python),
                        "execute": False,
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
