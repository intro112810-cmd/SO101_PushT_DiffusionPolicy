"""Audit camera registration in explicit fixture or governed physical mode."""

from __future__ import annotations

from so101_pusht_benchmark.sim_to_real.camera_audit_cli import run_camera_audit_cli


def main() -> int:
    """Default process has no production trust store and fails closed in physical mode."""
    return run_camera_audit_cli()


if __name__ == "__main__":
    raise SystemExit(main())
