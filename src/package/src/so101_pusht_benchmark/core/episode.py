"""Immutable episode alignment records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

from .contract import ContractError, Observation, PolicyInput, TimingContract
import itertools

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Episode", "EpisodeFrame"]


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    """A pre-action observation, action, and its exact timing record."""

    observation: Observation
    action: PolicyInput
    timing: TimingContract

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> EpisodeFrame:
        try:
            observation = Observation.parse(cast("Mapping[str, object]", value["observation"]))
            action = PolicyInput.parse(cast("Mapping[str, object]", value["action"]))
            frame_index = value["frame_index"]
            timestamp = value["timestamp"]
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not isinstance(timestamp, (int, float))
            ):
                raise ContractError("invalid frame timing types")
            timing = TimingContract.create(frame_index, float(timestamp))
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("malformed episode frame") from exc
        return cls(observation, action, timing)


@dataclass(frozen=True, slots=True)
class Episode:
    """A contiguous sequence of policy ticks."""

    frames: tuple[EpisodeFrame, ...]
    horizon: int = 300

    @classmethod
    def parse(cls, frames: tuple[EpisodeFrame, ...], horizon: int = 300) -> Episode:
        if not 0 < horizon <= 300 or not frames or len(frames) > horizon:
            raise ContractError("episode length or horizon is invalid")
        for previous, current in itertools.pairwise(frames):
            previous.timing.validate_next(current.timing)
        return cls(frames, horizon)
