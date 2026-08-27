"""Receipt-bound affine mapping for shadow observation assembly."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.real_shadow import validate_shadow_agent_pos
from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

Float32Vector = NDArray[np.float32]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_joint_mapping_receipt(path: Path) -> dict[str, object]:
    """Load one mapping receipt and reject malformed documents."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, "joint mapping receipt must be a mapping")
    return cast("dict[str, object]", raw)


def mapped_agent_pos_from_receipt(
    receipt: Mapping[str, object],
    *,
    receipt_path: Path,
) -> tuple[Float32Vector, str, dict[str, object]]:
    """Return float32[5] affine-mapped radians or fail closed on invalid elbow."""
    joints = receipt.get("joints")
    if not isinstance(joints, Mapping):
        raise RolloutViolation(RolloutCode.R_MISSING, "joint mapping receipt lacks joints")
    typed_joints = cast("Mapping[str, object]", joints)
    raw_blockers = receipt.get("blockers")
    blockers: list[str] = []
    if isinstance(raw_blockers, list):
        typed_blockers = cast("list[object]", raw_blockers)
        blockers = [str(item) for item in typed_blockers]
    elbow = typed_joints.get("elbow_flex")
    if isinstance(elbow, Mapping):
        typed_elbow = cast("Mapping[str, object]", elbow)
        if typed_elbow.get("valid") is False:
            raise RolloutViolation(
                RolloutCode.R_INVALID_ELBOW,
                "; ".join(str(item) for item in blockers) or "invalid elbow mapping",
            )
    mapped: list[float] = []
    for joint in JOINT_ORDER:
        entry = typed_joints.get(joint)
        if not isinstance(entry, Mapping):
            raise RolloutViolation(RolloutCode.R_MISSING, f"mapping receipt lacks {joint}")
        typed_entry = cast("Mapping[str, object]", entry)
        if typed_entry.get("valid") is not True:
            raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, f"{joint} mapping is invalid")
        value = typed_entry.get("mapped_q_radians")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RolloutViolation(RolloutCode.R_NONFINITE, f"{joint} mapped value is not numeric")
        mapped.append(float(value))
    agent_pos = validate_shadow_agent_pos(np.asarray(mapped, dtype=np.float32))
    evidence = {
        "joint_map_receipt": str(receipt_path),
        "joint_map_receipt_sha256": _sha256(receipt_path),
        "mapping_formula": receipt.get("mapping_formula"),
        "mapping_status": receipt.get("mapping_status"),
    }
    return agent_pos, "receipt_bound_affine_mapping", evidence
