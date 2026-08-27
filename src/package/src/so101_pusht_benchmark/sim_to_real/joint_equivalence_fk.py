"""Independent FK and approved-domain verification for joint corpus members."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from so101_pusht_benchmark.sim.scene import OVERLAY, UPSTREAM, mujoco

from .joint_equivalence_corpus import JointEquivalencePolicy, JointMember, unproven
from .joint_mapping import JOINT_ORDER
from .physical_ik_fk import build_joint_domains
from .read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
    require_read_only_acquisition_authority,
)


@dataclass(frozen=True, slots=True)
class FkVerification:
    """Independently recomputed FK residual summary."""

    residuals_m: tuple[float, ...]
    oracle: str = "pinned_mujoco_model_recomputed_from_raw_vectors"

    @property
    def maximum_residual_m(self) -> float:
        return max(self.residuals_m)


def derive_pinned_fk_positions(
    radian_vectors: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    """Derive tool XYZ from simulator vectors through the independent pinned model."""
    xml = OVERLAY.read_text(encoding="utf-8").replace(
        'file="so101_new_calib.xml"', f'file="{UPSTREAM}"'
    )
    assets = {path.name: path.read_bytes() for path in (UPSTREAM.parent / "assets").glob("*.stl")}
    try:
        model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    except ValueError as exc:
        raise unproven("independent pinned FK oracle is unavailable") from exc
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id == -1:
        raise unproven("independent pinned FK oracle lacks gripperframe")
    positions: list[tuple[float, float, float]] = []
    for radians in radian_vectors:
        if len(radians) != len(JOINT_ORDER):
            raise unproven("independent pinned FK oracle requires five joint radians")
        for index, value in enumerate(radians):
            data.qpos[index] = value
        data.qpos[len(JOINT_ORDER)] = 0.0
        mujoco.mj_forward(model, data)
        xyz = data.site_xpos[site_id]
        positions.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return positions


def _verify_residuals(members: Sequence[JointMember], max_fk_residual_m: float) -> FkVerification:
    oracle_positions = derive_pinned_fk_positions([member.radians for member in members])
    residuals = tuple(
        math.dist(member.measured_xyz, oracle_xyz)
        for member, oracle_xyz in zip(members, oracle_positions, strict=True)
    )
    if any(not math.isfinite(value) or value > max_fk_residual_m for value in residuals):
        raise unproven("independently recomputed FK residual exceeds approved policy")
    return FkVerification(residuals)


def verify_fk_read_only(
    members: Sequence[JointMember], authority: ProductionReadOnlyAcquisitionAuthority
) -> FkVerification:
    """Audit FK evidence without claiming joint domains or actuation eligibility."""
    approved = require_read_only_acquisition_authority(authority)
    return _verify_residuals(members, approved.kinematics.max_fk_residual_m)


def verify_joint_domains_and_fk(
    members: Sequence[JointMember], policy: JointEquivalencePolicy
) -> FkVerification:
    """Validate raw vectors against policy and recompute all FK residuals."""
    domains = build_joint_domains(policy)
    for member in members:
        for index, joint in enumerate(JOINT_ORDER):
            domain = domains[joint]
            if not domain.physical[0] <= member.degrees[index] <= domain.physical[1]:
                raise unproven("physical vector exceeds an approved joint domain")
            if not domain.mapped[0] <= member.radians[index] <= domain.mapped[1]:
                raise unproven("simulator vector exceeds an approved joint domain")
    return _verify_residuals(members, policy.kinematics.max_fk_residual_m)
