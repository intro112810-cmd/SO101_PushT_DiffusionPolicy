"""Thin re-export facade for the deterministic sim-to-real replay surface.

The implementation lives in the focused ``replay_*`` modules below. This
module preserves the import paths used by the replay CLI and its tests.
"""

from so101_pusht_benchmark.sim_to_real.replay_assemble import build_history
from so101_pusht_benchmark.sim_to_real.replay_policy import (
    fixture_policy_rng_seed,
    load_real_policy,
    run_fixture_policy,
)
from so101_pusht_benchmark.sim_to_real.replay_receipts import (
    parse_sample_document,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from so101_pusht_benchmark.sim_to_real.replay_serialize import (
    build_receipt,
    parse_inference_receipt,
    receipt_digest,
    stabilized_receipt,
    validate_inference_receipt,
    write_receipt,
)
from so101_pusht_benchmark.sim_to_real.replay_types import (
    CAMERA_REGISTRATION_DIGEST,
    FIXTURE_LINEAGE_ID,
    JOINT_EQUIVALENCE_DIGEST,
    PRODUCTION_LINEAGE_ID,
    Float32Vector,
    HistoryEvidence,
    HistoryStep,
    InferenceReceipt,
    PolicyRun,
    UInt8Image,
    validate_history_step_arrays,
)

__all__ = (
    "CAMERA_REGISTRATION_DIGEST",
    "FIXTURE_LINEAGE_ID",
    "JOINT_EQUIVALENCE_DIGEST",
    "PRODUCTION_LINEAGE_ID",
    "Float32Vector",
    "HistoryEvidence",
    "HistoryStep",
    "InferenceReceipt",
    "PolicyRun",
    "UInt8Image",
    "build_history",
    "build_receipt",
    "fixture_policy_rng_seed",
    "load_real_policy",
    "parse_inference_receipt",
    "parse_sample_document",
    "receipt_digest",
    "run_fixture_policy",
    "stabilized_receipt",
    "validate_camera_receipt",
    "validate_history_step_arrays",
    "validate_inference_receipt",
    "validate_joint_receipt",
    "validate_lineage_receipt",
    "write_receipt",
)
