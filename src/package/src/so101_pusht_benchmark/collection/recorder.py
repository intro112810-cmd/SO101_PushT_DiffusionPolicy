"""Injected-clock, fail-closed gamepad collection state machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np

from ..data.store import FrameRecord
from ..sim.env import PushTEnv
from .inputs import CollectionInput, SourceSample
from .types import CollectionConfig, RecorderState, fault_frame


class Clock(Protocol):
    def monotonic(self) -> float: ...


class AttemptStore(Protocol):
    def write_attempt(
        self, attempt_id: str, frames: list[FrameRecord], metadata: dict[str, object]
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class AttemptResult:
    accepted: bool
    failure_code: str | None
    frames: int
    attempt_path: Path


class PollingSource(Protocol):
    def poll(self) -> SourceSample: ...
    def close(self) -> None: ...


class Recorder:
    def __init__(
        self,
        env: PushTEnv,
        source: PollingSource,
        store: AttemptStore,
        config: CollectionConfig,
        clock: Clock,
        input_adapter: CollectionInput | None = None,
    ) -> None:
        self.env, self.source, self.store, self.config, self.clock = (
            env,
            source,
            store,
            config,
            clock,
        )
        self._input = input_adapter or CollectionInput.gamepad(config)
        self.state = RecorderState.DISCONNECTED
        self._last_fresh = clock.monotonic()
        self._buttons = {"success": -99, "stop": -99, "rerecord": -99}
        self._button_down = dict.fromkeys(self._buttons, False)
        self.last_frames: list[FrameRecord] = []

    def _abort(self, reason: str) -> str:
        self.env.abort_collection(reason)
        self.state = RecorderState.FAULT
        return reason

    def stop(self) -> None:
        self.env.stop_collection("software_stop")
        self.state = RecorderState.STOPPED

    def _edge(self, name: str, pressed: bool, frame: int) -> bool:
        if not pressed:
            self._button_down[name] = False
            return False
        if self._button_down[name] or frame - self._buttons[name] <= self._input.debounce_ticks:
            return False
        self._button_down[name], self._buttons[name] = True, frame
        return True

    def _arm(self, sample: object) -> bool:
        neutral = self._input.neutral(sample)
        if self.state is RecorderState.DISCONNECTED:
            self.state = RecorderState.NEUTRAL_REQUIRED
        if self.state is RecorderState.NEUTRAL_REQUIRED and neutral:
            self.state = RecorderState.ARMED
            return False
        return self.state is RecorderState.ARMED

    def record(self, seed: int, attempt_id: str, **options: object) -> AttemptResult:
        """Record an attempt; optional pacing and viewing stay outside frame storage."""
        _mode = options.pop("_mode", "human_gamepad")
        replacement_for = options.pop("replacement_for", None)
        max_ticks = options.pop("max_ticks", 300)
        before_tick = options.pop("before_tick", None)
        on_observation = options.pop("on_observation", None)
        if options:
            raise TypeError(f"unexpected record options: {sorted(options)}")
        if not isinstance(_mode, str):
            raise TypeError("_mode must be a string")
        if replacement_for is not None and not isinstance(replacement_for, str):
            raise TypeError("replacement_for must be a string or None")
        if not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if before_tick is not None and not callable(before_tick):
            raise TypeError("before_tick must be callable")
        if on_observation is not None and not callable(on_observation):
            raise TypeError("on_observation must be callable")
        typed_before_tick = cast("Callable[[], None] | None", before_tick)
        typed_on_observation = cast("Callable[[object], None] | None", on_observation)
        if self.state is RecorderState.STOPPED:
            raise RuntimeError("software stop is latched")
        observation, _ = self.env.reset(seed)
        observation = self._input.observe(self.env)
        self.state = RecorderState.DISCONNECTED
        frames: list[FrameRecord] = []
        mocap_target = self.env.scene.data.mocap_pos[0]
        target = (
            float(mocap_target[0]),
            float(mocap_target[1]),
            float(mocap_target[2]),
        )
        failure: str | None = None
        success_requested = False
        current: SourceSample | None = None

        def fault(reason: str) -> None:
            nonlocal failure
            raw_telemetry = self._input.raw(current) if current is not None else None
            frames.append(fault_frame(len(frames), observation, target, raw_telemetry, reason))
            failure = self._abort(reason)

        try:
            for tick in range(max_ticks):
                if typed_before_tick is not None:
                    typed_before_tick()
                if typed_on_observation is not None:
                    self._input.view(typed_on_observation, observation)
                current = None
                current, now = self.source.poll(), self.clock.monotonic()
                if current.connected and current.fresh:
                    self._last_fresh = now
                if not current.connected:
                    fault("disconnect")
                    break
                if now - self._last_fresh > self._input.stale_timeout_s:
                    fault("stale_input")
                    break
                buttons = {
                    name: self._edge(name, getattr(current, name), tick) for name in self._buttons
                }
                if sum(buttons.values()) > 1:
                    fault("button_conflict")
                    break
                if buttons["stop"]:
                    self.stop()
                    failure = "software_stop"
                    break
                if buttons["rerecord"]:
                    fault("rerecord")
                    break
                success_requested = success_requested or buttons["success"]
                if buttons["success"]:
                    self.stop_collection("operator_success_unverified")
                    failure = "operator_success_unverified"
                    break
                if not self._arm(current):
                    continue
                if not current.deadman:
                    if frames:
                        fault("deadman_released")
                        break
                    continue
                requested = self._input.requested(current, target)
                if requested is None:
                    fault("invalid_target")
                    break
                out = self.env.step(np.asarray(requested, dtype=np.float32))
                if out.terminated and "applied_target" not in out.info:
                    # Fault paths return info without applied_target (e.g.
                    # forbidden contact, terminal); record the fault instead
                    # of crashing on the missing key.
                    reason = str(out.info.get("fault") or "environment_terminal")
                    fault(reason)
                    break
                applied = cast("tuple[float, float, float]", out.info["applied_target"])
                raw_telemetry = self._input.raw(current)
                index = len(frames)
                frames.append(
                    FrameRecord(
                        index,
                        index / 10,
                        observation,
                        applied,
                        raw_telemetry,
                        requested,
                        {
                            **out.info,
                            "raw_axes": raw_telemetry,
                            "requested_target": requested,
                            "applied_action": applied,
                            "command_id": index * 2,
                            "frame_id": index * 2 + 1,
                            "observation_timestamp": index / 10,
                            "action_timestamp": index / 10,
                            "next_state_timestamp": (index + 1) / 10,
                        },
                        self._input.observe(self.env),
                        True,
                    )
                )
                observation = self._input.observe(self.env)
                target = applied
                if out.terminated:
                    failure = str(out.info.get("fault") or "environment_terminal")
                    break
        except KeyboardInterrupt:
            reason = "interrupted"
            if failure is None:
                frames.append(
                    fault_frame(
                        len(frames),
                        observation,
                        target,
                        self._input.raw(current) if current is not None else None,
                        reason,
                    )
                )
                failure = self._abort(reason)
        except Exception as exc:
            reason = f"loop_exception:{type(exc).__name__}"
            if failure is None:
                frames.append(
                    fault_frame(
                        len(frames),
                        observation,
                        target,
                        self._input.raw(current) if current is not None else None,
                        reason,
                    )
                )
                failure = self._abort(reason)
        if failure is None:
            failure = "operator_success_unverified" if success_requested else "coverage_not_met"
        is_public_device = self._input.physical_device
        metadata: dict[str, object] = {
            "attempt_id": attempt_id,
            "seed": seed,
            "task": "push_t",
            "mode": "synthetic_pipeline_probe"
            if _mode == "synthetic_pipeline_probe"
            else self._input.mode,
            "schema": self._input.schema,
            "physical_device": is_public_device,
            "device_provenance": {
                "adapter": self._input.provenance,
                "physical": is_public_device,
            },
            "synthetic": _mode == "synthetic_pipeline_probe",
            "success": False,
            "training_eligible": False,
            "failure_code": failure,
            "replacement_for": replacement_for,
            "operator_success_requested": success_requested,
            "timing": {"fps": 10, "substeps": 50, "timestamp": "frame_index/10"},
        }
        self.last_frames = frames
        return AttemptResult(
            False, failure, len(frames), self.store.write_attempt(attempt_id, frames, metadata)
        )

    def stop_collection(self, reason: str) -> None:
        self.env.stop_collection(reason)
        self.state = RecorderState.STOPPED


__all__ = ["AttemptResult", "Clock", "CollectionConfig", "Recorder", "RecorderState"]
