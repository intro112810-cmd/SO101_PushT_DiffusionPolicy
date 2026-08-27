from typing import Any

from so101_pusht_benchmark.training.identity import BundleIdentity


def load_policy(
    root: object,
    artifact_id: str,
    model: str,
) -> tuple[Any, BundleIdentity]:
    """Load a frozen policy and its identity through the shadow-inference path."""
    ...
