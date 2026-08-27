"""Deterministic policy selection and the fixture-only adapter.

The frozen policy load is attempted first and must fail closed when the
installed robomimic runtime lacks ``CropRandomizer``. A fixture-only lineage
may then select a deterministic adapter that consumes the validated observation
contract and emits a fixed eight-action raw chunk from a stable seed.
"""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np

from so101_pusht_benchmark.sim_to_real.replay_policy_loader import load_frozen_policy
from so101_pusht_benchmark.sim_to_real.replay_types import (
    ARTIFACT_ROOT,
    EXECUTED_ACTIONS,
    HistoryStep,
    PolicyRun,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.training.identity import BundleIdentity


def _missing_crop_randomizer(exc: BaseException) -> bool:
    cursor: BaseException | None = exc
    for _ in range(8):
        if cursor is None:
            break
        if isinstance(cursor, AttributeError) and "CropRandomizer" in str(cursor):
            return True
        cursor = getattr(cursor, "__cause__", None)
    return False


def fixture_policy_rng_seed(artifact_id: str, policy_seed: int) -> int:
    """Derive the same stable seed used by the fixture-only adapter."""
    encoded = f"fixture-sim-to-real-replay:{artifact_id}:{policy_seed}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def run_fixture_policy(
    history: tuple[HistoryStep, HistoryStep],
    *,
    seed: int,
) -> PolicyRun:
    """Consume the exact frozen-policy observation and emit a fixed chunk."""
    rng = np.random.default_rng(seed)
    states = np.stack([step.agent_pos for step in history], axis=0)
    frames = np.stack([step.checkpoint_image for step in history], axis=0)
    if frames.dtype != np.uint8 or frames.shape != (2, 96, 96, 3):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "fixture observation frame contract")
    if states.dtype != np.float32 or states.shape != (2, 5):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "fixture observation state contract")
    state_tokens = rng.integers(0, 1 << 32, size=(EXECUTED_ACTIONS, 2), dtype=np.int64)
    frame_tokens = rng.integers(0, 1 << 32, size=(EXECUTED_ACTIONS, 2), dtype=np.int64)
    tokens = state_tokens ^ frame_tokens
    raw = tokens.astype(np.float64) / float(1 << 32)
    scaled = raw * 2.0 - 1.0
    if not bool(np.isfinite(scaled).all()):
        raise RolloutViolation(
            RolloutCode.R_NONFINITE, "fixture action generation produced non-finite values"
        )
    if bool(np.any(scaled < -1.0)) or bool(np.any(scaled > 1.0)):
        raise RolloutViolation(
            RolloutCode.R_CLIPPING_REQUIRED, "fixture action generation exceeded [-1,1]"
        )
    observation_digest = hashlib.sha256()
    observation_digest.update(frames.tobytes())
    observation_digest.update(states.tobytes())
    conditioning_rng = np.random.default_rng(int.from_bytes(observation_digest.digest()[:8], "big"))
    conditioned = scaled + conditioning_rng.uniform(-1e-5, 1e-5, size=(EXECUTED_ACTIONS, 2))
    if bool(np.any(conditioned < -1.0)) or bool(np.any(conditioned > 1.0)):
        raise RolloutViolation(
            RolloutCode.R_CLIPPING_REQUIRED,
            "observation-conditioned fixture action exceeded [-1,1]",
        )
    actions = conditioned.astype(np.float32)
    return PolicyRun(
        actions=np.ascontiguousarray(actions),
        latency_seconds=0.0,
        policy="fixture_deterministic_adapter",
    )


def load_real_policy() -> tuple[object, object]:
    """Load the frozen policy through the package-local typed seam."""
    return load_frozen_policy(ARTIFACT_ROOT, "local-dp_cnn-recovered-v4-seed0", "dp_cnn")


def select_policy_run(
    lineage: dict[str, object],
    history: tuple[HistoryStep, HistoryStep],
    *,
    policy_seed: int,
) -> PolicyRun:
    if lineage["fixture_only"]:
        seed = fixture_policy_rng_seed(
            cast(str, lineage["artifact_id"]),
            policy_seed,
        )
        return run_fixture_policy(history, seed=seed)
    try:
        policy, identity = load_real_policy()
    except BaseException as exc:
        if _missing_crop_randomizer(exc) or (
            exc.__cause__ is not None and _missing_crop_randomizer(exc.__cause__)
        ):
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED,
                "robomimic.models.base_nets.CropRandomizer is missing from the "
                "frozen production runtime",
            ) from exc
        raise
    return _run_frozen_policy(policy, identity, history, policy_seed=policy_seed)


def _run_frozen_policy(
    policy: object,
    identity: object,
    history: tuple[HistoryStep, HistoryStep],
    *,
    policy_seed: int,
) -> PolicyRun:
    """Run the frozen policy and decode its exact eight-action raw chunk."""
    from collections import deque
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    import torch
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    from numpy.typing import NDArray

    from so101_pusht_benchmark.integrations.paper_baselines import runner as _runner
    from so101_pusht_benchmark.sim_to_real.replay_types import TensorToNumpy

    typed_policy = cast("BaseImagePolicy", policy)
    identity_model = cast("BundleIdentity", identity)
    observation_steps = identity_model.observation_steps
    executed_actions = identity_model.executed_actions
    observations: list[dict[str, NDArray[np.generic]]] = [
        {
            "cam_top": step.checkpoint_image,
            "agent_pos": step.agent_pos,
        }
        for step in history
    ]
    history_deque = deque(observations, maxlen=observation_steps)
    preserve = cast(
        "Callable[[BaseImagePolicy], AbstractContextManager[None]]",
        _runner.__dict__["_preserve_policy_rng"],
    )
    seed_rng = cast(
        "Callable[[BaseImagePolicy, int], int]",
        _runner.__dict__["_seed_policy_rng"],
    )
    with preserve(typed_policy):
        seed_rng(typed_policy, policy_seed)
        typed_policy.reset()
        with torch.no_grad():
            inputs = _runner.PaperBaselineRunner.policy_observation(history_deque, typed_policy)
            prediction = typed_policy.predict_action(inputs)
    action_tensor = prediction.get("action")
    if not isinstance(action_tensor, torch.Tensor):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "policy action tensor missing")
    raw = (
        cast(
            "TensorToNumpy",
            cast("object", action_tensor),
        )
        .detach()
        .cpu()
        .numpy(force=True)
    )
    chunk = np.asarray(raw, dtype=np.float32)
    if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 2:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "policy action shape")
    actions = np.asarray(chunk[0, :executed_actions], dtype=np.float32)
    return PolicyRun(actions=actions, latency_seconds=0.0, policy="frozen")
