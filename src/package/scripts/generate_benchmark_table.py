"""Render the paper-style four-model fixed-seed benchmark comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

from so101_pusht_benchmark.evaluation.professor_artifacts import (
    MODEL_ORDER,
    get_model_spec,
    validate_metrics_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        required=True,
        action="append",
        type=Path,
        help="Repeat exactly once per approved model.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.metrics) != len(MODEL_ORDER):
        raise ValueError("exactly four metrics receipts are required")
    by_model: dict[str, tuple[dict[str, object], object]] = {}
    for path in cast("list[Path]", args.metrics):
        metrics = cast("dict[str, object]", json.loads(path.read_text()))
        model = cast(str, metrics.get("model"))
        summary = validate_metrics_receipt(metrics, expected_model=model)
        if model in by_model:
            raise ValueError(f"duplicate metrics model: {model}")
        by_model[model] = (metrics, summary)
    if set(by_model) != set(MODEL_ORDER):
        raise ValueError("metrics receipts must cover all four approved models")
    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        metrics, _ = by_model[model]
        summary = validate_metrics_receipt(metrics, expected_model=model)
        rows.append(
            [
                get_model_spec(model).label,
                f"{summary.success_count}/100",
                f"{summary.success_rate:.3f}",
                f"{summary.mean_dxy:.4f}",
                f"{summary.mean_dyaw:.2f}",
                f"{float(metrics['eval/mean_duration_s']):.2f}",
            ]
        )
    columns = [
        "Model",
        "Success",
        "Rate ↑",
        "dxy (m) ↓",
        "dyaw (°) ↓",
        "Duration (s) ↓",
    ]
    figure, axis = plt.subplots(figsize=(12.5, 4.0), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#dbeafe"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.2)
    for column in range(len(columns)):
        table[(0, column)].set_text_props(weight="bold")
    best_row = max(range(len(rows)), key=lambda index: float(rows[index][2])) + 1
    for column in range(len(columns)):
        table[(best_row, column)].set_facecolor("#dcfce7")
        table[(best_row, column)].set_text_props(weight="bold")
    figure.suptitle(
        "Fixed-seed Push-T evaluation (100 seeds)",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.04,
        "Seeds 100000-100099 | 300-step cap | frozen SO-100 MuJoCo",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.savefig(args.output, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"published {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
