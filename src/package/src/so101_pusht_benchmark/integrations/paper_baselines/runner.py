"""Shared in-process MuJoCo runner for unchanged Stanford image policies.

Upstream symbol: BaseImageRunner at Stanford commit
5ba07ac6661db573af695b419a7947ecb704690f. Environment and metrics remain
project-owned; policy inference remains upstream-owned.
"""

from __future__ import annotations

import os

from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy

from .configs import PolicyNamespaceError

_from_numpy = cast("Callable[[NDArray[np.generic]], torch.Tensor]", torch.from_numpy)
# Upstream policies use NumPy's module-global RandomState API. These typed
# aliases are required to seed and restore that exact state; default_rng is a
# separate generator and cannot make those calls deterministic.
_numpy_get_state = cast("Callable[[], object]", vars(np.random)["get_state"])
_numpy_set_state = cast("Callable[[object], None]", vars(np.random)["set_state"])
_numpy_seed = cast("Callable[[int], None]", vars(np.random)["seed"])
_POLICY_SEED_DOMAIN = "pusht-so100-policy-rollout-v1"


def policy_seed(environment_seed: int) -> int:
    """Derive a stable 63-bit policy RNG seed from the environment seed."""
    encoded = f"{_POLICY_SEED_DOMAIN}:{environment_seed}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _torch_generators(policy: BaseImagePolicy) -> tuple[torch.Generator, ...]:
    generators: list[torch.Generator] = []
    seen: set[int] = set()
    for module in policy.modules():
        for value in vars(module).values():
            if isinstance(value, torch.Generator) and id(value) not in seen:
                seen.add(id(value))
                generators.append(value)
    return tuple(generators)


@contextmanager
def _preserve_policy_rng(policy: BaseImagePolicy) -> Generator[None, None, None]:
    python_state = random.getstate()
    numpy_state = _numpy_get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic_mode = torch.get_deterministic_debug_mode()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    generators = tuple((item, item.get_state()) for item in _torch_generators(policy))
    try:
        yield
    finally:
        random.setstate(python_state)
        _numpy_set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_deterministic_debug_mode(deterministic_mode)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        for generator, state in generators:
            generator.set_state(state)


def _seed_policy_rng(policy: BaseImagePolicy, environment_seed: int) -> int:
    derived = policy_seed(environment_seed)
    random.seed(derived)
    _numpy_seed(derived % (2**32))
    cast("Callable[[int], object]", torch.manual_seed)(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)
    torch.set_deterministic_debug_mode(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for generator in _torch_generators(policy):
        manual_seed = cast(
            "Callable[[torch.Generator, int], torch.Generator]",
            vars(type(generator))["manual_seed"],
        )
        manual_seed(generator, derived)
    return derived


class _StepResult(Protocol):
    observation: dict[str, NDArray[np.generic]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class _Environment(Protocol):
    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]: ...

    def step(self, action: object) -> _StepResult: ...

    def close(self) -> None: ...


def validate_native_runner_observation(
    value: object,
) -> dict[str, NDArray[np.generic]]:
    """Validate native HWC arrays before policy conversion or inference."""
    if not isinstance(value, dict):
        raise PolicyNamespaceError("native runner observation must be an ordered mapping")
    observation = cast("dict[str, object]", value)
    single_cam = os.environ.get("PUSHT_SINGLE_CAM") == "1"
    expected_keys = ("cam_top", "agent_pos") if single_cam else ("cam_top", "cam_side", "agent_pos")
    if tuple(observation) != expected_keys:
        raise PolicyNamespaceError("native runner observation keys/order mismatch")
    cameras = ("cam_top",) if single_cam else ("cam_top", "cam_side")
    for camera in cameras:
        image = observation[camera]
        if not isinstance(image, np.ndarray):
            raise PolicyNamespaceError(f"{camera} must be HWC uint8[{96 if single_cam else 224},{96 if single_cam else 224},3]")
        typed_image = cast("NDArray[np.generic]", image)
        expected_shape = (96, 96, 3) if single_cam else (224, 224, 3)
        if typed_image.shape != expected_shape or typed_image.dtype != np.dtype(np.uint8):
            raise PolicyNamespaceError(f"{camera} must be HWC uint8[{96 if single_cam else 224},{96 if single_cam else 224},3]")
    state = observation["agent_pos"]
    if not isinstance(state, np.ndarray):
        raise PolicyNamespaceError("agent_pos must be float32[5]")
    typed_state = cast("NDArray[np.generic]", state)
    if typed_state.shape != (5,) or typed_state.dtype != np.dtype(np.float32):
        raise PolicyNamespaceError("agent_pos must be float32[5]")
    return cast("dict[str, NDArray[np.generic]]", value)


class PaperBaselineRunner(BaseImageRunner):
    """Evaluate paired reset seeds and execute only each profile's chunk prefix."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        evaluation_seeds: tuple[int, ...] = tuple(range(100000, 100100)),
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        options: dict[str, object] | None = None,
    ) -> None:
        super().__init__(str(output_dir))
        values = {} if options is None else options
        if "env_factory" in values:
            raise PolicyNamespaceError("legacy env_factory option is forbidden")
        if set(values) - {"max_steps", "native_env_factory"}:
            raise ValueError("unknown runner option")
        raw_max_steps = values.get("max_steps", 300)
        if not isinstance(raw_max_steps, int) or raw_max_steps < 1:
            raise ValueError("runner seeds must be unique and horizons must be positive")
        max_steps: int = raw_max_steps
        env_factory = values.get("native_env_factory")
        if env_factory == "frozen":
            # Serialization-safe marker for the locked native environment; the
            # launcher injects it for full production, evaluation uses it too.
            from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht

            def frozen_factory() -> _Environment:
                return cast(
                    "_Environment",
                    cast("object", load_frozen_pusht(max_steps=max_steps)),
                )

            env_factory = frozen_factory
        if env_factory is None or not callable(env_factory):
            raise PolicyNamespaceError("native environment factory is required")
        factory_module = getattr(env_factory, "__module__", "")
        if factory_module.startswith("so101_pusht_benchmark.sim"):
            raise PolicyNamespaceError("legacy custom simulator is forbidden")
        if (
            len(evaluation_seeds) != len(set(evaluation_seeds))
            or n_obs_steps < 1
            or n_action_steps < 1
        ):
            raise ValueError("runner seeds must be unique and horizons must be positive")
        if not evaluation_seeds:
            # Full production defers native rollouts to the final evaluation
            # command; an empty seed list makes workspace.run() skip rollout.
            self.evaluation_seeds = ()
        self.evaluation_seeds = evaluation_seeds
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.env_factory = cast("Callable[[], _Environment]", env_factory)
        self._artifact_output_dir = Path(output_dir)

    @staticmethod
    def policy_observation(
        history: deque[dict[str, NDArray[np.generic]]], policy: BaseImagePolicy
    ) -> dict[str, torch.Tensor]:
        # strip HD auxiliary keys before validation
        history = [{k: v for k, v in item.items() if not k.startswith("_")} for item in history]
        native_history = [validate_native_runner_observation(item) for item in history]
        if isinstance(policy, RobomimicImagePolicy):
            # The pinned wrapper consumes x[:, 0] while its recurrent model
            # carries temporal state internally. Match the upstream recipe's
            # n_obs_steps=1 instead of forwarding the oldest item in the
            # dataset's 10-step training sequence.
            native_history = native_history[-1:]
        observations: dict[str, torch.Tensor] = {}
        single_cam = os.environ.get("PUSHT_SINGLE_CAM") == "1"
        cameras = ("cam_top",) if single_cam else ("cam_top", "cam_side")
        for camera in cameras:
            frames = np.stack([cast("NDArray[np.uint8]", item[camera]) for item in native_history])
            image = np.moveaxis(frames, -1, 1).astype(np.float32) / np.float32(255)
            observations[camera] = _from_numpy(image[None]).to(
                device=policy.device, dtype=policy.dtype
            )
        states = np.stack(
            [cast("NDArray[np.float32]", item["agent_pos"]) for item in native_history]
        )
        observations["agent_pos"] = _from_numpy(states[None]).to(
            device=policy.device, dtype=policy.dtype
        )
        return observations

    def _rollout(self, env: _Environment, policy: BaseImagePolicy, seed: int) -> dict[str, object]:
        with _preserve_policy_rng(policy):
            return self._rollout_seeded(env, policy, seed)

    def _rollout_seeded(
        self, env: _Environment, policy: BaseImagePolicy, seed: int
    ) -> dict[str, object]:
        reset_observation, _ = env.reset(seed=seed)
        observation = validate_native_runner_observation(reset_observation)
        history: deque[dict[str, NDArray[np.generic]]] = deque(
            (observation for _ in range(self.n_obs_steps)), maxlen=self.n_obs_steps
        )
        derived_policy_seed = _seed_policy_rng(policy, seed)
        policy.reset()
        steps = 0
        dxy = float("nan")
        dyaw = float("nan")
        terminated = False
        truncated = False
        while steps < self.max_steps and not terminated and not truncated:
            with torch.no_grad():
                prediction = policy.predict_action(self.policy_observation(history, policy))
            action_tensor = prediction.get("action")
            if not isinstance(action_tensor, torch.Tensor):
                raise TypeError("policy must return an action tensor")
            to_numpy = cast("Callable[[], NDArray[np.generic]]", action_tensor.detach().cpu().numpy)
            chunk = to_numpy()
            if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 2:
                raise RuntimeError("policy action must have shape [1, T, 2]")
            for raw_action in chunk[0, : self.n_action_steps]:
                if not isinstance(raw_action, np.ndarray):
                    raise TypeError("policy action must be exact float32[2]")
                typed_action = cast("NDArray[np.generic]", cast("object", raw_action))
                if typed_action.dtype != np.dtype(np.float32) or typed_action.shape != (2,):
                    raise ValueError("policy action must be exact float32[2]")
                action = cast("NDArray[np.float32]", cast("object", typed_action))
                if not bool(np.isfinite(action).all()):
                    raise ValueError("policy action must contain only finite values")
                if bool(np.any(action < -1.0)) or bool(np.any(action > 1.0)):
                    raise ValueError("policy action exceeds [-1,1] bounds; clipping is forbidden")
                result = env.step(action)
                next_observation = validate_native_runner_observation(result.observation)
                steps += 1
                history.append(next_observation)
                raw_dxy, raw_dyaw = result.info.get("dxy"), result.info.get("dyaw")
                if (
                    isinstance(raw_dxy, bool)
                    or not isinstance(raw_dxy, (int, float))
                    or isinstance(raw_dyaw, bool)
                    or not isinstance(raw_dyaw, (int, float))
                    or not np.isfinite(raw_dxy)
                    or not np.isfinite(raw_dyaw)
                ):
                    raise TypeError("environment dxy/dyaw telemetry is invalid")
                dxy, dyaw = float(raw_dxy), float(raw_dyaw)
                terminated = result.terminated
                truncated = result.truncated
                if terminated or truncated or steps >= self.max_steps:
                    break
        return {
            "seed": seed,
            "policy_seed": derived_policy_seed,
            "success": terminated,
            "dxy": dxy,
            "dyaw": dyaw,
            "duration_s": steps / 10,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated and not terminated,
        }

    def run(self, policy: BaseImagePolicy) -> dict[str, object]:
        candidate = self.env_factory()
        if not all(
            callable(getattr(candidate, member, None)) for member in ("reset", "step", "close")
        ):
            close = getattr(candidate, "close", None)
            if callable(close):
                close()
            raise PolicyNamespaceError("native environment adapter is invalid")
        if not self.evaluation_seeds:
            # Defer native rollouts entirely: no environment creation, so the
            # training process never touches MuJoCo/OpenGL.
            self._artifact_output_dir.mkdir(parents=True, exist_ok=True)
            return {"rollouts": [], "deferred": True}
        env = candidate
        try:
            rollouts = [self._rollout(env, policy, seed) for seed in self.evaluation_seeds]
        finally:
            env.close()
        result: dict[str, object] = {
            "eval/success_rate": float(np.mean([bool(item["success"]) for item in rollouts])),
            "eval/mean_dxy": float(np.mean([cast(float, item["dxy"]) for item in rollouts])),
            "eval/mean_dyaw": float(np.mean([cast(float, item["dyaw"]) for item in rollouts])),
            "eval/mean_duration_s": float(
                np.mean([cast(float, item["duration_s"]) for item in rollouts])
            ),
            "rollouts": rollouts,
        }
        self._artifact_output_dir.mkdir(parents=True, exist_ok=True)
        failures = [item for item in rollouts if item["success"] is not True]
        temporary = self._artifact_output_dir / ".failure_traces.json.tmp"
        try:
            temporary.write_text(
                json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(self._artifact_output_dir / "failure_traces.json")
        finally:
            temporary.unlink(missing_ok=True)
        return result
