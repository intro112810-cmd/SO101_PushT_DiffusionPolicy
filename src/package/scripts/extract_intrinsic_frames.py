#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "av==15.1.0",
#     "numpy>=1.24",
#     "opencv-python-headless==4.11.0.86",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv or pip install needed):
#      uv run scripts/extract_intrinsic_frames.py VIDEO --expected-sha256 SHA256
# 3. Or make executable and run:
#      chmod +x scripts/extract_intrinsic_frames.py && ./scripts/extract_intrinsic_frames.py --help
# ──────────────────

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from so101_pusht_benchmark.sim_to_real.intrinsic_extraction_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
