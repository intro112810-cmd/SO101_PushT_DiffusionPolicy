from __future__ import annotations

import sys

from so101_pusht_benchmark.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["export-inference-bundle", *sys.argv[1:]]))
