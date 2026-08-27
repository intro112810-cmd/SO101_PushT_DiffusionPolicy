"""Private typed threshold parsing and consistency checks."""

from __future__ import annotations

from .policy_schema import (
    JOINT_ORDER,
    SECTION_FIELDS,
    YamlMapping,
    YamlValue,
    boolean_value,
    mapping_value,
    numeric_range_value,
    polygon_value,
    positive_number,
    positive_integer,
    text_value,
    policy_unauthorized,
    vector_value,
)
from .policy_types import (
    AcknowledgementPolicy,
    BoundedRolloutBudget,
    CameraPolicy,
    CollisionPolicy,
    JointDomains,
    KinematicsPolicy,
    NumericRange,
    OperatorPolicy,
    PostStatePolicy,
    ProviderPolicy,
    SafetyThresholds,
    ShadowBudget,
    SingleStepBudget,
    SlewPolicy,
    TimingPolicy,
    WatchdogPolicy,
    WorkspacePolicy,
)

__all__: tuple[str, ...] = ()


def section_value(thresholds: YamlMapping, name: str) -> YamlMapping:
    return mapping_value(thresholds[name], name, SECTION_FIELDS[name])


def range_values(raw: YamlValue, label: str) -> tuple[NumericRange, ...]:
    values = mapping_value(raw, label, frozenset(JOINT_ORDER))
    return tuple(
        NumericRange(*numeric_range_value(values[name], f"{label}.{name}")) for name in JOINT_ORDER
    )


def parse_thresholds_internal(raw: YamlMapping) -> SafetyThresholds:
    value = raw["thresholds"]
    if not isinstance(value, dict):
        raise policy_unauthorized("thresholds fields are incomplete or unknown")
    expected = SECTION_FIELDS["thresholds"]
    if "collision" in value:
        expected = expected | {"collision"}
    thresholds = mapping_value(value, "thresholds", expected)
    workspace = section_value(thresholds, "workspace")
    joints = section_value(thresholds, "joint_domains")
    order = joints["joint_order"]
    if not isinstance(order, list) or tuple(order) != JOINT_ORDER:
        raise policy_unauthorized("joint_order must exactly match the five body joints")
    timing = section_value(thresholds, "timing")
    camera = section_value(thresholds, "camera")
    kinematics = section_value(thresholds, "kinematics")
    collision = section_value(thresholds, "collision") if "collision" in thresholds else None
    slew = section_value(thresholds, "slew")
    provider = section_value(thresholds, "provider")
    watchdog = section_value(thresholds, "watchdog")
    acknowledgement = section_value(thresholds, "acknowledgement")
    post_state = section_value(thresholds, "post_state")
    shadow = section_value(thresholds, "shadow")
    single = section_value(thresholds, "single_step")
    bounded = section_value(thresholds, "bounded_rollout")
    operator = section_value(thresholds, "operator")

    parsed_timing = TimingPolicy(
        positive_number(timing["sample_max_age_seconds"], "timing.sample_max_age_seconds"),
        positive_number(timing["sample_max_skew_seconds"], "timing.sample_max_skew_seconds"),
        positive_number(timing["max_policy_age_seconds"], "timing.max_policy_age_seconds"),
        positive_number(
            timing["authorization_max_age_seconds"], "timing.authorization_max_age_seconds"
        ),
        positive_number(timing["authorization_ttl_seconds"], "timing.authorization_ttl_seconds"),
    )
    parsed_camera = CameraPolicy(
        positive_number(camera["max_reprojection_error_px"], "camera.max_reprojection_error_px"),
        positive_integer(camera["min_correspondences"], "camera.min_correspondences"),
        positive_number(
            camera["max_correspondence_error_px"], "camera.max_correspondence_error_px"
        ),
    )
    parsed_provider = ProviderPolicy(
        boolean_value(provider["exact_goal_required"], "provider.exact_goal_required"),
        positive_number(provider["max_abs_error_degrees"], "provider.max_abs_error_degrees"),
    )
    parsed_watchdog = WatchdogPolicy(
        positive_number(watchdog["timeout_seconds"], "watchdog.timeout_seconds")
    )
    parsed_ack = AcknowledgementPolicy(
        boolean_value(acknowledgement["required"], "acknowledgement.required"),
        positive_number(acknowledgement["timeout_seconds"], "acknowledgement.timeout_seconds"),
        positive_number(
            acknowledgement["max_position_error_degrees"],
            "acknowledgement.max_position_error_degrees",
        ),
    )
    parsed_operator = OperatorPolicy(
        boolean_value(operator["deadman_required"], "operator.deadman_required"),
        boolean_value(operator["stop_required"], "operator.stop_required"),
        text_value(operator["stop_behavior"], "operator.stop_behavior"),
        boolean_value(operator["acknowledgement_required"], "operator.acknowledgement_required"),
    )
    orientation = vector_value(
        workspace["tool_orientation_rpy_rad"],
        "workspace.tool_orientation_rpy_rad",
        3,
    )
    result = SafetyThresholds(
        WorkspacePolicy(
            polygon_value(workspace["polygon_xy_m"]),
            positive_number(workspace["contact_z_m"], "workspace.contact_z_m"),
            (orientation[0], orientation[1], orientation[2]),
        ),
        JointDomains(
            JOINT_ORDER,
            range_values(joints["physical_degrees"], "physical_degrees"),
            range_values(joints["mapped_radians"], "mapped_radians"),
        ),
        parsed_timing,
        parsed_camera,
        KinematicsPolicy(
            positive_number(kinematics["max_fk_residual_m"], "kinematics.max_fk_residual_m"),
            positive_number(kinematics["max_ik_residual_m"], "kinematics.max_ik_residual_m"),
            positive_number(
                kinematics["min_singularity_metric"], "kinematics.min_singularity_metric"
            ),
            positive_number(
                kinematics["max_branch_delta_degrees"],
                "kinematics.max_branch_delta_degrees",
            ),
        ),
        None
        if collision is None
        else CollisionPolicy(
            positive_number(collision["minimum_clearance_m"], "collision.minimum_clearance_m"),
            positive_number(
                collision["max_joint_step_radians"], "collision.max_joint_step_radians"
            ),
            positive_integer(collision["max_path_samples"], "collision.max_path_samples"),
        ),
        SlewPolicy(
            positive_number(slew["max_cartesian_delta_m"], "slew.max_cartesian_delta_m"),
            positive_number(slew["max_joint_delta_degrees"], "slew.max_joint_delta_degrees"),
        ),
        parsed_provider,
        parsed_watchdog,
        parsed_ack,
        PostStatePolicy(
            positive_number(post_state["max_age_seconds"], "post_state.max_age_seconds"),
            positive_number(
                post_state["max_tracking_error_degrees"],
                "post_state.max_tracking_error_degrees",
            ),
        ),
        ShadowBudget(
            positive_integer(shadow["min_cycles"], "shadow.min_cycles"),
            positive_number(
                shadow["max_cycle_latency_seconds"], "shadow.max_cycle_latency_seconds"
            ),
            positive_integer(shadow["max_error_count"], "shadow.max_error_count"),
        ),
        SingleStepBudget(positive_integer(single["max_commands"], "single_step.max_commands")),
        BoundedRolloutBudget(
            positive_integer(bounded["max_commands"], "bounded_rollout.max_commands"),
            positive_number(
                bounded["max_duration_seconds"], "bounded_rollout.max_duration_seconds"
            ),
            positive_number(bounded["max_path_length_m"], "bounded_rollout.max_path_length_m"),
            positive_integer(bounded["max_error_count"], "bounded_rollout.max_error_count"),
        ),
        parsed_operator,
    )
    if parsed_timing.sample_max_skew_seconds > parsed_timing.sample_max_age_seconds:
        raise policy_unauthorized("sample skew cannot exceed sample freshness")
    if parsed_timing.authorization_ttl_seconds > parsed_timing.authorization_max_age_seconds:
        raise policy_unauthorized("authorization TTL cannot exceed maximum age")
    if parsed_camera.max_reprojection_error_px > parsed_camera.max_correspondence_error_px:
        raise policy_unauthorized("reprojection limit cannot exceed correspondence limit")
    if parsed_ack.timeout_seconds >= parsed_watchdog.timeout_seconds:
        raise policy_unauthorized("acknowledgement timeout must precede watchdog timeout")
    if result.single_step.max_commands != 1:
        raise policy_unauthorized("single-step command budget must be exactly 1")
    mandatory = (
        parsed_operator.deadman_required,
        parsed_operator.stop_required,
        parsed_operator.acknowledgement_required,
        parsed_ack.required,
        parsed_provider.exact_goal_required,
    )
    if not all(mandatory) or parsed_operator.stop_behavior != "latch_hold":
        raise policy_unauthorized(
            "deadman, stop, acknowledgement, and provider equality are mandatory"
        )
    return result
