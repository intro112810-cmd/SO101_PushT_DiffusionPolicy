"""Paper-runtime identity checks for executable model operations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)


class PaperRuntimeError(RuntimeError):
    """Raised outside the pinned isolated paper runtime."""


def assert_paper_runtime() -> None:
    if sys.version_info[:2] != (3, 10) or torch.__version__ != "2.7.1+cu128":
        raise PaperRuntimeError("model operations require Python 3.10 and Torch 2.7.1+cu128")
    module = sys.modules[DiffusionUnetHybridImagePolicy.__module__]
    origin_value = getattr(module, "__file__", None)
    if not isinstance(origin_value, str):
        raise PaperRuntimeError("DP-CNN upstream module has no file origin")
    origin = Path(origin_value).resolve()
    expected = Path(__file__).resolve().parents[5] / (
        "04_experiments/so101_pusht_benchmark/cache/upstream/stanford"
    )
    if expected.resolve() not in origin.parents:
        raise PaperRuntimeError(f"DP-CNN import is not from pristine Stanford source: {origin}")
    try:
        __import__("lerobot")
    except ImportError:
        return
    raise PaperRuntimeError("LeRobot must be absent from the isolated paper runtime")
