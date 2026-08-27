"""Canonical inference receipt serialization.

Owns building the receipt from a selected policy run, its digest, boundary
parsing, and atomic idempotent writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np

from so101_pusht_benchmark.sim_to_real.ledger_io import LedgerDocument
from so101_pusht_benchmark.sim_to_real.policy_canonical import JsonValue
from so101_pusht_benchmark.sim_to_real.replay_policy import select_policy_run
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
    validate_receipt_path,
)
from so101_pusht_benchmark.sim_to_real.replay_receipts import (
    canonical,
    require_action_rows,
    require_digest,
    require_float,
    require_mapping,
    sha256_bytes,
)
from so101_pusht_benchmark.sim_to_real.replay_types import (
    FIXTURE_LINEAGE_ID,
    PRODUCTION_LINEAGE_ID,
    RECEIPT_FIELDS,
    HistoryStep,
    InferenceReceipt,
    round_trip_float,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.secure_io import atomic_write_new, unlink_owned_leaf


def _canonical_document(receipt: LedgerDocument) -> bytes:
    data = {key: receipt[key] for key in RECEIPT_FIELDS}
    return canonical(data)


def build_receipt(
    history: tuple[HistoryStep, HistoryStep],
    *,
    lineage: LedgerDocument,
    joint_digest: str,
    camera_digest: str,
    policy_seed: int,
) -> InferenceReceipt:
    """Run the selected policy path and produce the canonical receipt."""
    run = select_policy_run(lineage, history, policy_seed=policy_seed)
    payload: LedgerDocument = {
        "schema": 1,
        "mode": "sim_to_real_first_rollout_replay",
        "policy": run.policy,
        "policy_attempt": "frozen"
        if lineage["artifact_id"] == PRODUCTION_LINEAGE_ID
        else "fixture",
        "artifact_id": lineage["artifact_id"],
        "lineage_authority_digest": lineage["authority_digest"],
        "lineage_digest": lineage["lineage_digest"],
        "joint_digest": joint_digest,
        "camera_digest": camera_digest,
        "sample_ids": [step.sample_id for step in history],
        "sample_digests": [step.sample_digest for step in history],
        "camera_sha256s": [step.camera_sha256 for step in history],
        "agent_pos_sha256s": [step.agent_pos_sha256 for step in history],
        "action_chunk_float32_2d": [[float(v) for v in row] for row in run.actions.tolist()],
        "seed": policy_seed,
        "latency_seconds": round_trip_float(run.latency_seconds),
        "deployment_valid": False,
        "hardware_actuation": False,
        "crop_randomizer_missing": False,
    }
    return InferenceReceipt(
        schema=1,
        mode="sim_to_real_first_rollout_replay",
        policy=run.policy,
        policy_attempt=cast(Literal["frozen", "fixture"], payload["policy_attempt"]),
        artifact_id=cast("str", lineage["artifact_id"]),
        lineage_authority_digest=cast("str", lineage["authority_digest"]),
        lineage_digest=cast("str", lineage["lineage_digest"]),
        joint_digest=joint_digest,
        camera_digest=camera_digest,
        sample_ids=tuple(step.sample_id for step in history),
        sample_digests=tuple(step.sample_digest for step in history),
        camera_sha256s=tuple(step.camera_sha256 for step in history),
        agent_pos_sha256s=tuple(step.agent_pos_sha256 for step in history),
        action_chunk=run.actions,
        seed=policy_seed,
        latency_seconds=run.latency_seconds,
        deployment_valid=False,
        hardware_actuation=False,
        crop_randomizer_missing=False,
    )


def receipt_digest(receipt: InferenceReceipt) -> str:
    """Return the canonical inference digest bound to every receipt field."""
    return sha256_bytes(_canonical_document(receipt.to_document()))


def parse_inference_receipt(raw: LedgerDocument) -> InferenceReceipt:
    """Parse one receipt from the canonical wire mapping."""
    document = require_mapping(raw, "inference receipt")
    if set(document) != set(RECEIPT_FIELDS) | {"inference_digest"}:
        raise RolloutViolation(RolloutCode.R_MISSING, "inference receipt fields")
    if document.get("schema") != 1:
        raise RolloutViolation(RolloutCode.R_MISSING, "inference receipt schema")
    policy = document.get("policy")
    if policy not in {"frozen", "fixture_deterministic_adapter"}:
        raise RolloutViolation(RolloutCode.R_MISSING, "inference policy is unknown")
    attempt = document.get("policy_attempt")
    if attempt not in {"frozen", "fixture"}:
        raise RolloutViolation(RolloutCode.R_MISSING, "inference policy attempt is unknown")
    if not isinstance(policy, str) or not isinstance(attempt, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "inference policy/attempt must be strings")
    raw_sample_ids = document.get("sample_ids")
    raw_sample_digests = document.get("sample_digests")
    raw_camera_sha256s = document.get("camera_sha256s")
    raw_agent_pos_sha256s = document.get("agent_pos_sha256s")
    inventory = cast(
        "tuple[list[JsonValue], list[JsonValue], list[JsonValue], list[JsonValue]]",
        (raw_sample_ids, raw_sample_digests, raw_camera_sha256s, raw_agent_pos_sha256s),
    )
    if not all(
        len(value) == 2 and all(isinstance(item, str) for item in value) for value in inventory
    ):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "receipt history inventory")
    sample_ids = cast("list[str]", raw_sample_ids)
    sample_digests = cast("list[str]", raw_sample_digests)
    camera_sha256s = cast("list[str]", raw_camera_sha256s)
    agent_pos_sha256s = cast("list[str]", raw_agent_pos_sha256s)
    action_raw = document.get("action_chunk_float32_2d")
    action_rows = require_action_rows(action_raw, "action chunk")
    action_chunk = np.asarray(action_rows, dtype=np.float32)
    if bool(np.any(action_chunk < -1.0)) or bool(np.any(action_chunk > 1.0)):
        raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "action chunk bounds")
    seed = document.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "seed must be an integer")
    latency = require_float(document.get("latency_seconds"), "latency_seconds")
    for key in (
        "lineage_authority_digest",
        "lineage_digest",
        "joint_digest",
        "camera_digest",
    ):
        require_digest(document.get(key), key)
    if document.get("artifact_id") not in {FIXTURE_LINEAGE_ID, PRODUCTION_LINEAGE_ID}:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "artifact_id identity drift")
    for key in ("deployment_valid", "hardware_actuation", "crop_randomizer_missing"):
        if document.get(key) is not False:
            raise RolloutViolation(RolloutCode.R_MISSING, f"{key} must be false")
    receipt = InferenceReceipt(
        schema=1,
        mode=str(document["mode"]),
        policy=cast("Literal['frozen', 'fixture_deterministic_adapter']", policy),
        policy_attempt=cast("Literal['frozen', 'fixture']", attempt),
        artifact_id=str(document["artifact_id"]),
        lineage_authority_digest=str(document["lineage_authority_digest"]),
        lineage_digest=str(document["lineage_digest"]),
        joint_digest=str(document["joint_digest"]),
        camera_digest=str(document["camera_digest"]),
        sample_ids=tuple(sample_ids),
        sample_digests=tuple(sample_digests),
        camera_sha256s=tuple(camera_sha256s),
        agent_pos_sha256s=tuple(agent_pos_sha256s),
        action_chunk=action_chunk,
        seed=seed,
        latency_seconds=latency,
        deployment_valid=False,
        hardware_actuation=False,
        crop_randomizer_missing=False,
    )
    if receipt_digest(receipt) != require_digest(
        document.get("inference_digest"), "inference_digest"
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "inference receipt digest")
    return receipt


def validate_inference_receipt(raw: LedgerDocument) -> None:
    """Raise on any canonical receipt drift."""
    parse_inference_receipt(raw)


def _norm_document(raw: LedgerDocument) -> LedgerDocument:
    document = dict(raw)
    rows = document.get("action_chunk_float32_2d")
    if isinstance(rows, list):
        typed_rows = cast("list[JsonValue]", rows)
        document["action_chunk_float32_2d"] = [
            [
                round_trip_float(float(cast("float", value)))
                for value in cast("list[JsonValue]", row)
            ]
            for row in typed_rows
            if isinstance(row, list)
        ]
    if "latency_seconds" in document:
        document["latency_seconds"] = round_trip_float(
            float(cast("float", document["latency_seconds"]))
        )
    return document


def stabilized_receipt(raw: LedgerDocument) -> str:
    """Return the byte-stable digest shared by all equivalent receipts."""
    return sha256_bytes(canonical(_norm_document(raw)))


def write_receipt(receipt: InferenceReceipt, output: Path) -> Path:
    """Write the canonical receipt atomically and idempotently.

    Running the same deterministic replay against the same output path twice
    must remain a success. A prior byte-identical receipt is accepted as-is;
    a prior receipt with different content is rejected rather than overwritten.
    """
    document = receipt.to_document()
    document["inference_digest"] = receipt_digest(receipt)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    production = receipt.policy == "frozen"
    output = validate_receipt_path(output, production=production)
    prepare_receipt_directory(output.parent, production=production)
    location = locate_receipt_path(output)
    identity = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        encoded,
        temporary=f".{output.name}.tmp",
        accept_identical=True,
    )
    accepted = False
    try:
        validate_receipt_identity(location, production=production)
        accepted = True
        return location.lexical
    finally:
        if not accepted:
            unlink_owned_leaf(identity)
