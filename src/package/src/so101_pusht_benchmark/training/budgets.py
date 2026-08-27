from __future__ import annotations


APPROVED_OPTIMIZER_UPDATES = {
    "dp_cnn": 1_794_000,
    "dp_transformer": 1_794_000,
    "ibc": 100_000,
    "lstm_gmm": 300_000,
}

# Local single-camera 96x96 exploratory budget (server 12-run contract unchanged).
# DP models drop to ~400k updates (paper-faithful scale); IBC/LSTM keep the
# original budgets because they were already small.
LOCAL_OPTIMIZER_UPDATES = {
    "dp_cnn": 400_000,
    "dp_transformer": 400_000,
    "ibc": 100_000,
    "lstm_gmm": 300_000,
}


def approved_optimizer_updates(model: str) -> int:
    try:
        return APPROVED_OPTIMIZER_UPDATES[model]
    except KeyError as exc:
        raise ValueError(f"unknown approved model budget: {model}") from exc


def local_optimizer_updates(model: str) -> int:
    """Return the local exploratory budget (PUSHT_LOCAL_BUDGET=1) or the approved one."""
    import os

    if os.environ.get("PUSHT_LOCAL_BUDGET") == "1":
        try:
            return LOCAL_OPTIMIZER_UPDATES[model]
        except KeyError as exc:
            raise ValueError(f"unknown local model budget: {model}") from exc
    return approved_optimizer_updates(model)
