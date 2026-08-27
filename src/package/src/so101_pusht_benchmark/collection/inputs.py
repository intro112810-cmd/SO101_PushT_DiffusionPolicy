"""Per-source input adapters for the collection recorder.

The recorder's ``record()`` loop is source-agnostic: it polls a sample, runs
an arming gate, derives an absolute XYZ request, stores raw telemetry, and
pumps the live viewer.  ``CollectionInput`` is the seam that supplies the
per-source differences; the default (gamepad) adapter reproduces the v1
behavior byte-for-byte, and ``MouseKeyboardInput`` swaps in the schema-3
mouse semantics: left-hold deadman, absolute pixel-to-task targets, and raw
topdown observations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..control.paper_view import PaperView
from ..sim.env import PushTEnv
from .types import CollectionConfig

Observation = dict[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class _PaperState:
    """Pose bundle matching the PaperView state protocol."""

    t_x: float
    t_y: float
    t_yaw: float
    pusher_x: float
    pusher_y: float


class SourceSample(Protocol):
    """Read-only attribute surface the record loop needs on every sample."""

    @property
    def deadman(self) -> bool: ...

    @property
    def connected(self) -> bool: ...

    @property
    def fresh(self) -> bool: ...

    @property
    def success(self) -> bool: ...

    @property
    def stop(self) -> bool: ...

    @property
    def rerecord(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class CollectionInput:
    """How a polled source sample becomes a requested target and observation.

    All callables receive the sample produced by ``source.poll()``.  The
    default construction reproduces the v1 gamepad path exactly: axes-neutral
    arming, incremental ``CollectionConfig``-driven requests, raw axes
    telemetry, and front-camera observations.
    """

    mode: str
    physical_device: bool
    schema: int
    provenance: str
    stale_timeout_s: float
    debounce_ticks: int
    neutral: Callable[[Any], bool]
    requested: Callable[[Any, tuple[float, float, float]], tuple[float, float, float] | None]
    raw: Callable[[Any], Any]
    observe: Callable[[PushTEnv], Observation]
    view: Callable[[Callable[[Any], None], Observation], None]

    @classmethod
    def gamepad(cls, config: CollectionConfig, *, mode: str = "human_gamepad") -> CollectionInput:
        """Return the v1 gamepad adapter (identical behavior to the old loop)."""

        def neutral(sample: Any) -> bool:
            axes = getattr(sample, "axes", (0.0, 0.0, 0.0))
            return axes == (0.0, 0.0, 0.0) and not bool(getattr(sample, "deadman", True))

        def requested(
            sample: Any, previous: tuple[float, float, float]
        ) -> tuple[float, float, float] | None:
            return gamepad_request(
                config, cast("tuple[float, float, float]", sample.axes), previous
            )

        def raw(sample: Any) -> Any:
            return getattr(sample, "axes", None)

        def observe(env: PushTEnv) -> Observation:
            return env.observe()

        def view(on_observation: Callable[[Any], None], observation: Observation) -> None:
            on_observation(
                cast(
                    "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]",
                    observation["observation.images.front"],
                )
            )

        return cls(
            mode=mode,
            physical_device=True,
            schema=1,
            provenance="lerobot_public_gamepad",
            stale_timeout_s=config.stale_timeout_s,
            debounce_ticks=config.debounce_ticks,
            neutral=neutral,
            requested=requested,
            raw=raw,
            observe=observe,
            view=view,
        )

    @classmethod
    def mouse(
        cls,
        *,
        stale_timeout_s: float,
        debounce_ticks: int,
        contact_z_m: float,
        clearance_z_m: float,
        bounds_x: tuple[float, float],
        bounds_y: tuple[float, float],
    ) -> CollectionInput:
        """Return the schema-3 mouse adapter.

        The mouse source already computes absolute XYZ targets in the task
        frame (``sample.target``); the recorder simply forwards them.  ``None``
        targets (outside bounds, deadman released, or focus lost) are rejected
        fail-closed by the loop.  Recorded observations are the raw topdown
        camera plus the 15-float state; the viewer callback receives the full
        observation dict so the CLI can render the two-pane display.
        """

        def neutral(sample: Any) -> bool:
            return not bool(getattr(sample, "deadman", True))

        def requested(
            sample: Any, _previous: tuple[float, float, float]
        ) -> tuple[float, float, float] | None:
            target = getattr(sample, "target", None)
            if target is None:
                return None
            tx, ty, tz = cast("tuple[float, float, float]", target)
            if not (
                bounds_x[0] <= tx <= bounds_x[1]
                and bounds_y[0] <= ty <= bounds_y[1]
                and contact_z_m <= tz <= clearance_z_m
            ):
                return None
            return (float(tx), float(ty), float(tz))

        def raw(sample: Any) -> Any:
            return getattr(sample, "target", None)

        def observe(env: PushTEnv) -> Observation:
            base = env.observe()
            paper = PaperView(bounds_x=bounds_x, bounds_y=bounds_y)
            tx, ty, tyaw, px, py = env.paper_state
            return {
                "observation.images.topdown": paper.render(
                    _PaperState(tx, ty, tyaw, px, py), size=96
                ),
                "observation.state": base["observation.state"],
            }

        def view(on_observation: Callable[[Any], None], observation: Observation) -> None:
            on_observation(observation)

        return cls(
            mode="human_mouse_keyboard",
            physical_device=False,
            schema=3,
            provenance="mouse_keyboard_topdown_v3",
            stale_timeout_s=stale_timeout_s,
            debounce_ticks=debounce_ticks,
            neutral=neutral,
            requested=requested,
            raw=raw,
            observe=observe,
            view=view,
        )


def gamepad_request(
    config: CollectionConfig, raw: tuple[float, float, float], previous: tuple[float, float, float]
) -> tuple[float, float, float]:
    axes = np.asarray(raw, dtype=np.float32)
    if axes.shape != (3,) or not bool(np.isfinite(axes).all()) or bool(np.abs(axes).max() > 1):
        raise ValueError("raw gamepad axes must be a finite XYZ tuple in [-1,1]")
    axes[np.abs(axes) < config.deadzone] = 0
    z_target = previous[2] + float(axes[config.z_axis]) * config.z_meters_per_tick
    return (
        previous[0] + float(axes[0]) * config.xy_meters_per_tick,
        previous[1] + float(axes[1]) * config.xy_meters_per_tick,
        max(config.z_contact_m, min(config.z_approach_m, z_target)),
    )


__all__ = ["CollectionInput", "gamepad_request"]
