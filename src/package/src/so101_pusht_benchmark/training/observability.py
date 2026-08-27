from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    model: str
    training_seed: int
    dataset_digest: str
    split_digest: str
    runtime_digest: str
    configured_budget: dict[str, object]
    host: str
    systemd_unit: str
    started_at: str = ""


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    value = asdict(metadata)
    value["schema"] = "pusht-training-run-metadata-v1"
    value["started_at"] = metadata.started_at or _utc_now()
    _atomic_json(path, value)


class ResourceSampler:
    def __init__(
        self,
        path: Path,
        *,
        sample: Callable[[], Mapping[str, int | float]],
    ) -> None:
        self._path = path
        self._sample = sample
        self._started = time.monotonic()

    def capture(self, *, global_step: int, epoch: int) -> None:
        value: dict[str, object] = {
            "schema": "pusht-training-resource-sample-v1",
            "captured_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - self._started,
            "global_step": global_step,
            "epoch": epoch,
            **self._sample(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
