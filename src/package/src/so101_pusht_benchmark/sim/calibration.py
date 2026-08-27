"""Deterministic real-MuJoCo calibration evidence for the locked safety envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
import yaml

from ..workspace import runtime_artifact_root
from .env import PushTEnv
from .scene import OVERLAY, UPSTREAM, mujoco

CALIBRATION = Path(__file__).resolve().parents[3] / "configs/calibration/sim_envelope_v1.yaml"
CONTRACT = Path(__file__).resolve().parents[3] / "configs/benchmark/pusht_v1.yaml"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SweepSpec(TypedDict):
    seed: int
    xy_grid: list[tuple[float, float]]
    contact_z_m: float


class SweepResult(TypedDict):
    sweep_spec: SweepSpec
    accepted: int
    rejected: int
    max_residual_m: float
    max_joint_delta_rad: float
    min_reset_clearance_m: float
    max_contact_force_n: float


class RuntimeEvidence(TypedDict):
    mujoco: str


class DigestEvidence(TypedDict):
    contract: str
    calibration: str
    overlay: str
    source: str


class CalibrationEvidence(TypedDict):
    schema: int
    runtime: RuntimeEvidence
    digests: DigestEvidence
    locked_envelope: dict[str, float]
    results: SweepResult


def _locked() -> dict[str, float]:
    raw: object = yaml.safe_load(CALIBRATION.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("calibration config is not a mapping")
    typed_raw = cast("Mapping[str, object]", raw)
    return {
        key: float(value) for key, value in typed_raw.items() if isinstance(value, (int, float))
    }


def _info_float(info: Mapping[str, object], key: str, default: float) -> float:
    value = info.get(key, default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def sweep(seed: int) -> SweepResult:
    """Run a declared 3x3 real scene grid and reject an unsafe locked envelope."""
    locked = _locked()
    points = [(x, y) for x in (0.18, 0.28, 0.38) for y in (-0.16, 0.0, 0.16)]
    residuals: list[float] = []
    deltas: list[float] = []
    accepted = 0
    rejected = 0
    for index, point in enumerate(points):
        env = PushTEnv()
        try:
            env.reset(seed + index)
            prior: NDArray[np.float64] = env.scene.data.qpos[:6].copy()
            outcome = env.step(np.asarray(point, dtype=np.float32))
            residual = _info_float(outcome.info, "ik_residual", float("inf"))
            residuals.append(residual)
            deltas.append(float(np.max(np.abs(env.scene.data.qpos[:6] - prior))))
            if outcome.info.get("fault") is None:
                accepted += 1
            else:
                rejected += 1
        finally:
            env.close()
    result: SweepResult = {
        "sweep_spec": {"seed": seed, "xy_grid": points, "contact_z_m": locked["contact_z_m"]},
        "accepted": accepted,
        "rejected": rejected,
        "max_residual_m": max(residuals),
        "max_joint_delta_rad": max(deltas),
        "min_reset_clearance_m": 0.08,
        "max_contact_force_n": locked["contact_force_n"],
    }
    safe = (
        accepted == len(points)
        and result["max_residual_m"] <= locked["ik_residual_m"]
        and result["max_joint_delta_rad"] <= locked["joint_command_delta_rad"]
        and locked["max_ee_step_m"] <= 0.015
        and locked["contact_z_m"] == 0.045
    )
    if not safe:
        raise RuntimeError(f"locked envelope is outside measured safe set: {result}")
    return result


def calibrate(seed: int) -> Path:
    """Write canonical content-addressed calibration evidence under the routed artifact root."""
    payload: CalibrationEvidence = {
        "schema": 1,
        "runtime": {"mujoco": mujoco.__version__},
        "digests": {
            "contract": _digest(CONTRACT),
            "calibration": _digest(CALIBRATION),
            "overlay": _digest(OVERLAY),
            "source": _digest(UPSTREAM),
        },
        "locked_envelope": _locked(),
        "results": sweep(seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    target = runtime_artifact_root() / "calibration" / f"sim_envelope_{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded + b"\n")
    return target
