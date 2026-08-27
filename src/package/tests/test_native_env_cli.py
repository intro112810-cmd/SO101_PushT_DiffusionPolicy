from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import cast

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
import pytest

from so101_pusht_benchmark import native_cli
from so101_pusht_benchmark.evaluation import frozen_env
from so101_pusht_benchmark.evaluation.frozen_env import FrozenStep


@dataclass
class _Counters:
    environment_constructions: int = 0
    environment_steps: int = 0
    outputs: int = 0
    actions: list[NDArray[np.float32]] = field(default_factory=list)


class _Environment:
    def __init__(self, counters: _Counters) -> None:
        self._counters = counters

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        return _observation(), {"seed": seed}

    def step(self, action: object) -> FrozenStep:
        checked = cast("NDArray[np.float32]", action)
        self._counters.environment_steps += 1
        self._counters.actions.append(checked)
        return FrozenStep(_observation(), 0.0, False, False, {"dxy": 0.0, "dyaw": 0.0})

    def close(self) -> None:
        pass


def _observation() -> dict[str, NDArray[np.generic]]:
    return {
        "cam_top": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_side": np.zeros((224, 224, 3), dtype=np.uint8),
        "agent_pos": np.zeros(5, dtype=np.float32),
    }


@dataclass(frozen=True)
class _InjectedParser:
    evidence: Path

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        del argv
        return argparse.Namespace(
            command="native-env-smoke",
            steps=1,
            action=[True, 0.0],
            evidence=self.evidence,
            seed=100000,
        )


def _install_fakes(monkeypatch: pytest.MonkeyPatch, counters: _Counters) -> None:
    monkeypatch.setattr(native_cli, "native_runtime_report", dict)

    def load_frozen_pusht(*, max_steps: int = 300) -> _Environment:
        del max_steps
        counters.environment_constructions += 1
        return _Environment(counters)

    def write_output(*args: object, **kwargs: object) -> None:
        del args, kwargs
        counters.outputs += 1

    monkeypatch.setattr(frozen_env, "load_frozen_pusht", load_frozen_pusht)
    monkeypatch.setattr(iio, "imwrite", write_output)


@pytest.mark.parametrize("outside", [1.00000001, -1.00000001])
def test_cli_rejects_float32_rounding_bypass_before_environment_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outside: float,
) -> None:
    counters = _Counters()
    _install_fakes(monkeypatch, counters)
    evidence = tmp_path / "must-not-exist"

    result = native_cli.main(
        [
            "native-env-smoke",
            "--action",
            repr(outside),
            "0",
            "--evidence",
            str(evidence),
        ]
    )

    assert result == 1
    assert "[-1,1] bounds; clipping is forbidden" in capsys.readouterr().out
    assert counters == _Counters()
    assert not evidence.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_injected_raw_bool_is_rejected_before_environment_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    counters = _Counters()
    _install_fakes(monkeypatch, counters)
    evidence = tmp_path / "must-not-exist"
    parser = _InjectedParser(evidence)

    monkeypatch.setattr(native_cli, "command_parser", lambda: parser)

    assert native_cli.main(["native-env-smoke"]) == 1
    assert "bool is forbidden" in capsys.readouterr().out
    assert counters == _Counters()
    assert not evidence.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize(
    "case",
    [
        (("nan", "0"), "finite", False),
        (("inf", "0"), "finite", False),
        (("True", "0"), "invalid float", True),
        (("0",), "float32[2]", False),
        (("0", "0", "0"), "float32[2]", False),
    ],
)
def test_raw_native_action_rejection_has_zero_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: tuple[tuple[str, ...], str, bool],
) -> None:
    counters = _Counters()
    _install_fakes(monkeypatch, counters)
    evidence = tmp_path / "must-not-exist"
    action, expected_error, parser_rejection = case
    argv = ["native-env-smoke", "--action", *action, "--evidence", str(evidence)]

    if parser_rejection:
        with pytest.raises(SystemExit, match="2"):
            native_cli.main(argv)
    else:
        assert native_cli.main(argv) == 1
    captured = capsys.readouterr()
    assert expected_error in captured.out + captured.err
    assert counters == _Counters()
    assert not evidence.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize(
    "action",
    [
        ("-1.0", "1.0"),
        (repr(math.nextafter(-1.0, 0.0)), repr(math.nextafter(1.0, 0.0))),
    ],
)
def test_raw_native_action_boundaries_convert_only_after_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: tuple[str, str],
) -> None:
    counters = _Counters()
    _install_fakes(monkeypatch, counters)
    evidence = tmp_path / "evidence"

    result = native_cli.main(["native-env-smoke", "--action", *action, "--evidence", str(evidence)])

    assert result == 0
    assert counters.environment_constructions == 1
    assert counters.environment_steps == 1
    assert counters.outputs == 2
    assert len(counters.actions) == 1
    assert counters.actions[0].dtype == np.dtype(np.float32)
    assert counters.actions[0].shape == (2,)
    assert evidence.is_dir()
