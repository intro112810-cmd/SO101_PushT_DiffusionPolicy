"""Shared identities, filenames, and validation for evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast


MODEL_ORDER = ("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")
EVALUATION_SEEDS = tuple(range(100000, 100100))


@dataclass(frozen=True, slots=True)
class ReportModelSpec:
    model: str
    label: str
    artifact_id: str


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    model: str
    success_count: int
    success_rate: float
    mean_dxy: float
    mean_dyaw: float
    optimizer_updates: int


_SPECS = {
    "dp_cnn": ReportModelSpec(
        "dp_cnn",
        "DP-CNN",
        "local-dp_cnn-recovered-v3-seed0",
    ),
    "dp_transformer": ReportModelSpec(
        "dp_transformer",
        "DP-Transformer",
        "local-dp_transformer-seed0",
    ),
    "ibc": ReportModelSpec("ibc", "IBC", "local-ibc-seed0"),
    "lstm_gmm": ReportModelSpec(
        "lstm_gmm",
        "LSTM-GMM",
        "local-lstm_gmm-seed0",
    ),
}


def get_model_spec(model: str) -> ReportModelSpec:
    """Return one of the four frozen evaluation model identities."""
    try:
        return _SPECS[model]
    except KeyError as error:
        raise ValueError(f"unsupported evaluation model: {model}") from error


def reel_filename(model: str, *, success: bool) -> str:
    """Return the final model-specific reel filename."""
    get_model_spec(model)
    outcome = "success" if success else "failure"
    return f"2026-08-20_{model}_{outcome}_three_view_120s_4x_hold2s.mp4"


def figure_filename(model: str) -> str:
    """Return the final fixed-state trajectory figure filename."""
    get_model_spec(model)
    return f"2026-08-20_{model}_paper_fixed_state_40steps.png"


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def validate_metrics_receipt(
    value: object,
    *,
    expected_model: str,
) -> MetricsSummary:
    """Validate one exact 100-seed evaluation receipt and summarize it."""
    get_model_spec(expected_model)
    if not isinstance(value, dict):
        raise TypeError("metrics receipt must be a mapping")
    receipt = cast("dict[str, object]", value)
    if receipt.get("model") != expected_model:
        raise ValueError("metrics model identity mismatch")
    if receipt.get("evaluation_seeds") != list(EVALUATION_SEEDS):
        raise ValueError("evaluation seeds must be exactly 100000-100099")
    raw_rollouts = receipt.get("rollouts")
    if not isinstance(raw_rollouts, list):
        raise TypeError("metrics rollouts must be a list")
    rollouts = cast("list[object]", raw_rollouts)
    if len(rollouts) != 100:
        raise ValueError("metrics receipt must contain exactly 100 rollouts")
    seen: set[int] = set()
    success_count = 0
    for item in rollouts:
        if not isinstance(item, dict):
            raise TypeError("rollout must be a mapping")
        rollout = cast("dict[str, object]", item)
        seed = rollout.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("rollout seed must be an integer")
        seen.add(seed)
        success = rollout.get("success")
        if not isinstance(success, bool):
            raise TypeError("rollout success must be boolean")
        success_count += int(success)
        _number(rollout.get("dxy"), name="rollout dxy")
        _number(rollout.get("dyaw"), name="rollout dyaw")
    if seen != set(EVALUATION_SEEDS):
        raise ValueError("rollout seeds must be unique and complete")
    success_rate = _number(receipt.get("eval/success_rate"), name="success rate")
    if not math.isclose(success_rate, success_count / 100, abs_tol=1e-12):
        raise ValueError("success rate disagrees with rollouts")
    updates = receipt.get("optimizer_updates", 0)
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("optimizer updates must be a positive integer")
    return MetricsSummary(
        model=expected_model,
        success_count=success_count,
        success_rate=success_rate,
        mean_dxy=_number(receipt.get("eval/mean_dxy"), name="mean dxy"),
        mean_dyaw=_number(receipt.get("eval/mean_dyaw"), name="mean dyaw"),
        optimizer_updates=updates,
    )
