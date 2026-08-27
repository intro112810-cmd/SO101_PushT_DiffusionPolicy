from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from so101_pusht_benchmark.data.store import FrameRecord


@dataclass(frozen=True)
class Attempt:
    metadata: dict[str, object]
    frames: list[FrameRecord]


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.attempts: dict[str, Attempt] = {}
        root.mkdir(parents=True, exist_ok=True)

    def write_attempt(
        self, attempt_id: str, frames: Sequence[FrameRecord], metadata: dict[str, object]
    ) -> Path:
        self.attempts[attempt_id] = Attempt(metadata, list(frames))
        path = self.root / f"{attempt_id}.json"
        path.write_text("raw attempt retained", encoding="utf-8")
        return path
