from __future__ import annotations

from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1]
SURFACE = (
    BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/__init__.py",
    BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/contracts.py",
    BENCHMARK / "src/so101_pusht_benchmark/sim_to_real/preview.py",
    BENCHMARK / "scripts/run_sim_to_real_dry_run.py",
)
FORBIDDEN = (
    "send_action",
    "sync_write",
    "Goal_Position",
    "enable_torque",
    "Torque_Enable",
    "configure_motors",
    "lerobot.teleoperators",
)


def test_sim_to_real_surface_exists_without_motor_write_symbols() -> None:
    missing = [str(path.relative_to(BENCHMARK)) for path in SURFACE if not path.is_file()]

    assert not missing, f"missing isolated dry-run surface: {missing}"
    source = "\n".join(path.read_text(encoding="utf-8") for path in SURFACE)
    assert all(symbol not in source for symbol in FORBIDDEN)
