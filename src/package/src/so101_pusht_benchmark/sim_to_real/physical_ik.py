"""Owned no-clipping physical IK planner.

Consumes a receipted Cartesian target plus a fresh measured body-degree seed,
binds the Todo 8 joint-equivalence receipt digest, solves five body joints
with damped least squares against the pinned physical MuJoCo model, and emits
a frozen body-only proposal. Every joint that would leave an approved domain
rejects instead of clipping; the gripper never appears in any payload.
"""

from __future__ import annotations

import math

import numpy as np
from so101_pusht_benchmark.sim.scene import Scene
from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import (
    MuJoCoWorkspace,
    build_joint_domains,
    check_elbow_domain,
    degrees_to_radians,
    forward_site,
    parse_body_degrees,
    parse_cartesian_target,
    radians_to_degrees,
    validate_joint_equivalence_digest,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_collision import (
    pinned_model_digest,
    swept_collision_proof,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_proposal import (
    PhysicalIKProposal,
    physical_ik_proposal_hash,
    round_trip_float,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import (
    SceneObjectPoseReceipt,
    apply_scene_object_pose,
)
from so101_pusht_benchmark.sim_to_real.physical_ik_solve import (
    SolveOutcome,
    SolveParams,
    solve_dls,
)
from so101_pusht_benchmark.sim_to_real.policy_types import (
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt


class PhysicalIKPlanner:
    """Branch-stable, five-joint planner bound to the pinned physical model."""

    def __init__(self, workspace: MuJoCoWorkspace) -> None:
        self._workspace = workspace

    @property
    def collision_workspace(self) -> MuJoCoWorkspace:
        """Expose the read-only model seam used by direct collision probes."""
        return self._workspace

    def plan(
        self,
        *,
        target: CartesianProposalReceipt,
        seed_degrees: object,
        joint_equivalence_digest: str,
        policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
        scene_pose: SceneObjectPoseReceipt | None = None,
    ) -> PhysicalIKProposal:
        """Plan one body-only proposal or raise the exact RED rejection code."""
        apply_scene_object_pose(self._workspace, scene_pose)
        validate_joint_equivalence_digest(joint_equivalence_digest)
        if target.policy_digest != policy.canonical_digest:
            raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "IK policy digest drift")
        domains = build_joint_domains(policy)
        seed = parse_body_degrees(seed_degrees)
        if any("gripper" in key for key in domains):
            raise RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "gripper has no planner domain")
        check_elbow_domain(seed, domains)
        degrees_to_radians(seed, domains)
        target_xyz = parse_cartesian_target(target)
        mapped_domains = {
            joint: (domain.mapped[0], domain.mapped[1]) for joint, domain in domains.items()
        }
        seed_radians = degrees_to_radians(seed, domains)
        params = SolveParams(
            max_iterations=64,
            tolerance_m=policy.kinematics.max_ik_residual_m,
            damping=0.01,
            min_singularity_metric=policy.kinematics.min_singularity_metric,
        )
        outcome = solve_dls(target_xyz, seed_radians, mapped_domains, self._workspace, params)
        if not outcome.converged:
            assert outcome.code is not None
            raise RolloutViolation(outcome.code, _rejection_detail(outcome))
        solution_degrees = radians_to_degrees(outcome.radians, domains)
        check_elbow_domain(solution_degrees, domains)
        branch_delta = max(
            abs(solution - seed_value)
            for solution, seed_value in zip(solution_degrees, seed, strict=True)
        )
        if branch_delta > policy.kinematics.max_branch_delta_degrees:
            raise RolloutViolation(
                RolloutCode.R_BRANCH_DISCONTINUITY,
                (
                    f"branch jump {branch_delta:.6f} degrees exceeds policy; "
                    f"solution_degrees={solution_degrees}"
                ),
            )
        final_residual = _final_fk_residual(
            self._workspace,
            outcome.radians,
            target_xyz,
            policy.kinematics.max_fk_residual_m,
        )
        if final_residual > policy.kinematics.max_fk_residual_m:
            raise RolloutViolation(
                RolloutCode.R_IK_UNREACHABLE,
                f"FK residual {final_residual:.6f} m exceeds policy",
            )
        collision_samples = swept_collision_proof(
            self._workspace, seed_radians, outcome.radians, policy.collision, scene_pose
        )
        swept_path = tuple(sample.site_xyz for sample in collision_samples)
        model_digest = pinned_model_digest()
        document: dict[str, object] = {
            "schema": 1,
            "mode": "physical_body_only_ik_proposal",
            "joint_order": list(JOINT_ORDER),
            "body_degrees": list(solution_degrees),
            "fk_residual_m": round_trip_float(final_residual),
            "singularity_metric": round_trip_float(outcome.singular_value),
            "branch_delta_degrees": round_trip_float(branch_delta),
            "swept_path": [list(point) for point in swept_path],
            "collision_samples": [sample.to_document() for sample in collision_samples],
            "model_digest": model_digest,
            "policy_digest": policy.canonical_digest,
            "scene_pose_digest": collision_samples[0].pose_digest,
            "obstacle_transforms": [
                [name, list(transform)]
                for name, transform in collision_samples[0].obstacle_transforms
            ],
            "clipping_performed": False,
            "gripper_present": False,
            "joint_equivalence_digest": joint_equivalence_digest,
        }
        proposal_hash = physical_ik_proposal_hash(document)
        document["proposal_hash"] = proposal_hash
        return PhysicalIKProposal(
            body_degrees=solution_degrees,
            fk_residual_m=final_residual,
            singularity_metric=outcome.singular_value,
            branch_delta_degrees=branch_delta,
            swept_path=swept_path,
            clipping_performed=False,
            gripper_present=False,
            joint_equivalence_digest=joint_equivalence_digest,
            proposal_hash=proposal_hash,
            collision_samples=collision_samples,
            model_digest=model_digest,
            policy_digest=policy.canonical_digest,
            scene_pose_digest=collision_samples[0].pose_digest,
            obstacle_transforms=collision_samples[0].obstacle_transforms,
        )


def _rejection_detail(outcome: SolveOutcome) -> str:
    return (
        f"{outcome.code.value if outcome.code else 'unknown'}: "
        f"residual {outcome.residual_m:.6f} m, "
        f"singularity {outcome.singular_value:.6f}"
    )


def _final_fk_residual(
    workspace: MuJoCoWorkspace,
    radians: tuple[float, float, float, float, float],
    target: tuple[float, float, float],
    max_fk_residual_m: float,
) -> float:
    position = forward_site(workspace, radians)
    residual = float(np.linalg.norm(np.asarray(target) - position))
    if not math.isfinite(residual):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "FK residual must be finite")
    if residual > max_fk_residual_m:
        raise RolloutViolation(
            RolloutCode.R_IK_UNREACHABLE, f"FK residual {residual:.6f} m exceeds policy"
        )
    return residual


def build_physical_ik_planner() -> PhysicalIKPlanner:
    """Construct the planner over the pinned physical MuJoCo scene."""
    scene = Scene()
    mujoco = scene.mujoco
    site_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id == -1:
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "gripperframe missing")
    workspace = MuJoCoWorkspace(
        scene,
        int(site_id),
        np.empty((3, scene.model.nv), dtype=np.float64),
        np.empty((3, scene.model.nv), dtype=np.float64),
    )
    return PhysicalIKPlanner(workspace)
