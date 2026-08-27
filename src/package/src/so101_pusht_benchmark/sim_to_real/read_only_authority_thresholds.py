"""Explicit owner-supplied values for read-only authority signing."""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ("AcquisitionThresholdInputs",)


class AcquisitionThresholdError(ValueError):
    """An owner-supplied signing threshold is invalid."""


@dataclass(frozen=True, slots=True)
class AcquisitionThresholdInputs:
    """Values written into a signed authority without production defaults."""

    camera_readiness_timeout_seconds: float
    joint_connect_timeout_seconds: float
    sample_pair_completion_timeout_seconds: float
    shutdown_grace_seconds: float
    camera_priming_frame_count: int
    accepted_sample_pair_count: int
    sample_max_age_seconds: float
    sample_max_skew_seconds: float
    max_fk_residual_m: float
    max_reprojection_error_px: float
    max_correspondence_error_px: float
    min_correspondences: int

    def __post_init__(self) -> None:
        """Reject invalid values before any signing key is generated."""
        values = (
            self.camera_readiness_timeout_seconds,
            self.joint_connect_timeout_seconds,
            self.sample_pair_completion_timeout_seconds,
            self.shutdown_grace_seconds,
            self.sample_max_age_seconds,
            self.sample_max_skew_seconds,
            self.max_fk_residual_m,
            self.max_reprojection_error_px,
            self.max_correspondence_error_px,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise AcquisitionThresholdError("all read-only thresholds must be finite and positive")
        if self.camera_priming_frame_count != 1 or self.accepted_sample_pair_count != 2:
            raise AcquisitionThresholdError(
                "capture cardinality must be exactly one priming frame and two pairs"
            )
        if isinstance(self.min_correspondences, bool) or self.min_correspondences <= 0:
            raise AcquisitionThresholdError("minimum correspondences must be a positive integer")
