from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
UPSTREAM_SOURCE = (
    PACKAGE_ROOT.parents[1] / "05_references/external_repos/pushT-so100/src/env_human_ee.py"
)
UPSTREAM_HELPER = UPSTREAM_SOURCE.with_name("helper.py")


@dataclass
class FakeDataset:
    frames: list[dict[str, object]] = field(default_factory=list)
    save_calls: int = 0
    fail_save: bool = False
    clear_calls: int = 0
    finalize_calls: int = 0
    fail_finalize: bool = False
    fail_clear: bool = False
    add_calls: int = 0
    fail_add_at: int | None = None
    add_error: BaseException | None = None
    save_error: BaseException | None = None

    @property
    def num_episodes(self) -> int:
        return self.save_calls

    def add_frame(self, frame: dict[str, object]) -> None:
        self.add_calls += 1
        if self.add_calls == self.fail_add_at:
            if self.add_error is not None:
                raise self.add_error
            raise RuntimeError("frame write failed")
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.save_calls += 1
        if self.save_error is not None:
            raise self.save_error
        if self.fail_save:
            raise RuntimeError("disk full")

    def clear_episode_buffer(self, *, delete_images: bool = True) -> None:
        assert delete_images is True
        self.clear_calls += 1
        if self.fail_clear:
            raise RuntimeError("buffer cleanup failed")

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize:
            raise RuntimeError("finalize failed")


@dataclass
class FakeJoystick:
    name: str
    axes: int
    buttons: int
    initialized: bool = False

    def init(self) -> None:
        self.initialized = True

    def get_name(self) -> str:
        return self.name

    def get_numaxes(self) -> int:
        return self.axes

    def get_numbuttons(self) -> int:
        return self.buttons


def _extract_function(name: str, namespace: dict[str, object]) -> Callable[..., object]:
    source = UPSTREAM_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(UPSTREAM_SOURCE))
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1, f"expected one upstream {name} function"
    compiled = compile(
        ast.fix_missing_locations(
            ast.Module(body=[cast("ast.stmt", functions[0])], type_ignores=[])
        ),
        filename=str(UPSTREAM_SOURCE),
        mode="exec",
    )
    function_codes = [
        value
        for value in compiled.co_consts
        if isinstance(value, CodeType) and value.co_name == name
    ]
    assert len(function_codes) == 1
    return cast("Callable[..., object]", FunctionType(function_codes[0], namespace))


def _frame(action_x: float) -> dict[str, object]:
    return {
        "cam_top": np.zeros((2, 2, 3), dtype=np.uint8),
        "cam_side": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": np.zeros(5, dtype=np.float64),
        "mocap_pose_2d": np.array([action_x, 0.0], dtype=np.float32),
    }


def _all_abs(value: object) -> bool:
    array = cast("NDArray[np.float32]", value)
    return bool(np.all(np.abs(array) < 0.0001))


def _ignore(*_args: object) -> None:
    return None


def _ignore_path(_path: Path) -> None:
    return None


def _save_harness(
    dataset: FakeDataset | None,
    *,
    fail_init: bool = False,
    init_error: BaseException | None = None,
) -> tuple[Callable[..., object], list[Path], list[Path]]:
    init_calls: list[Path] = []
    quarantine_calls: list[Path] = []

    def init_lerobot_dataset(*, repo_path: Path) -> FakeDataset:
        init_calls.append(repo_path)
        if init_error is not None:
            raise init_error
        if fail_init:
            raise RuntimeError("dataset creation failed")
        return FakeDataset()

    namespace: dict[str, object] = {
        "dataset": dataset,
        "init_lerobot_dataset": init_lerobot_dataset,
        "REPO_ID": Path("/unused"),
        "np": np,
        "np_allabs": _all_abs,
        "TOLERANCE": 0.0001,
        "logger": SimpleNamespace(info=_ignore, warning=_ignore),
        "quarantine_failed_dataset": quarantine_calls.append,
        "reject_dataset_tree_symlinks": _ignore_path,
    }
    return _extract_function("save_successful_episode", namespace), init_calls, quarantine_calls


def test_unsuccessful_episode_is_rejected_before_dataset_creation() -> None:
    save, init_calls, quarantine_calls = _save_harness(None)

    assert save([_frame(0.0)], succeeded=False) is False
    assert init_calls == []
    assert quarantine_calls == []


def test_successful_episode_saves_synchronously_and_keeps_final_frame() -> None:
    dataset = FakeDataset()
    save, _, quarantine_calls = _save_harness(dataset)

    assert save([_frame(0.0), _frame(0.2), _frame(0.4)], succeeded=True) is True
    assert [
        cast("NDArray[np.float32]", frame["action"]).tolist() for frame in dataset.frames
    ] == [
        [0.0, 0.0],
        pytest.approx([0.2, 0.0]),
        pytest.approx([0.4, 0.0]),
    ]
    assert dataset.save_calls == 1
    assert quarantine_calls == []


def test_episode_save_failure_propagates_to_collection_process() -> None:
    dataset = FakeDataset(fail_save=True)
    save, _, quarantine_calls = _save_harness(dataset)

    with pytest.raises(RuntimeError, match="disk full"):
        save([_frame(0.0)], succeeded=True)

    assert dataset.clear_calls == 1
    assert dataset.finalize_calls == 1
    assert quarantine_calls == [Path("/unused")]


def test_dataset_creation_failure_quarantines_partial_repository() -> None:
    save, init_calls, quarantine_calls = _save_harness(None, fail_init=True)

    with pytest.raises(RuntimeError, match="dataset creation failed"):
        save([_frame(0.0)], succeeded=True)

    assert init_calls == [Path("/unused")]
    assert quarantine_calls == [Path("/unused")]


def test_add_frame_failure_finalizes_and_quarantines_partial_repository() -> None:
    dataset = FakeDataset(fail_add_at=2)
    save, _, quarantine_calls = _save_harness(dataset)

    with pytest.raises(RuntimeError, match="frame write failed"):
        save([_frame(0.0), _frame(0.2)], succeeded=True)

    assert dataset.clear_calls == 1
    assert dataset.finalize_calls == 1
    assert quarantine_calls == [Path("/unused")]


@pytest.mark.parametrize(
    ("dataset", "cleanup_message"),
    [
        (FakeDataset(fail_save=True, fail_clear=True), "buffer cleanup failed"),
        (FakeDataset(fail_save=True, fail_finalize=True), "finalize failed"),
    ],
)
def test_cleanup_failures_still_quarantine_and_preserve_persistence_cause(
    dataset: FakeDataset,
    cleanup_message: str,
) -> None:
    save, _, quarantine_calls = _save_harness(dataset)

    with pytest.raises(
        RuntimeError,
        match=rf"episode persistence failed: disk full; cleanup failed: {cleanup_message}",
    ) as captured:
        save([_frame(0.0)], succeeded=True)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "disk full"
    assert quarantine_calls == [Path("/unused")]


@pytest.mark.parametrize("stage", ["init", "add", "save"])
def test_operator_interrupt_always_quarantines_and_is_re_raised(stage: str) -> None:
    if stage == "init":
        dataset = None
        save, _, quarantine_calls = _save_harness(
            dataset,
            init_error=KeyboardInterrupt(),
        )
    else:
        dataset = FakeDataset(
            fail_add_at=1 if stage == "add" else None,
            add_error=KeyboardInterrupt() if stage == "add" else None,
            save_error=KeyboardInterrupt() if stage == "save" else None,
        )
        save, _, quarantine_calls = _save_harness(dataset)

    with pytest.raises(KeyboardInterrupt):
        save([_frame(0.0)], succeeded=True)

    if dataset is not None:
        assert dataset.clear_calls == 1
        assert dataset.finalize_calls == 1
    assert quarantine_calls == [Path("/unused")]


def test_finalize_dataset_calls_explicit_api_and_surfaces_errors() -> None:
    dataset = FakeDataset()
    namespace: dict[str, object] = {"dataset": dataset}
    finalize = _extract_function("finalize_dataset", namespace)

    assert finalize() is True
    assert dataset.finalize_calls == 1
    assert namespace["dataset"] is None

    failed = FakeDataset(fail_finalize=True)
    namespace["dataset"] = failed
    with pytest.raises(RuntimeError, match="finalize failed"):
        finalize()
    assert failed.finalize_calls == 1
    assert namespace["dataset"] is failed


def test_quarantine_failed_dataset_atomically_removes_canonical_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    dataset_root.mkdir()
    marker = dataset_root / "partial.parquet"
    marker.write_bytes(b"partial")
    existing_failed = tmp_path / ".episodes.failed"
    existing_failed.mkdir()
    (existing_failed / "older").write_bytes(b"keep")
    namespace: dict[str, object] = {
        "os": os,
        "uuid4": lambda: SimpleNamespace(hex="deadbeef"),
    }
    quarantine = _extract_function("quarantine_failed_dataset", namespace)

    failed_root = cast("Path", quarantine(dataset_root))

    assert failed_root == tmp_path / ".episodes.failed-deadbeef"
    assert not dataset_root.exists()
    assert (failed_root / "partial.parquet").read_bytes() == b"partial"
    assert (existing_failed / "older").read_bytes() == b"keep"


def test_child_rejects_dataset_descendant_symlink_before_persistence(tmp_path: Path) -> None:
    dataset_root = tmp_path / "episodes"
    dataset_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (dataset_root / "images").symlink_to(outside, target_is_directory=True)
    namespace: dict[str, object] = {"os": os}
    reject_symlinks = _extract_function("reject_dataset_tree_symlinks", namespace)

    with pytest.raises(RuntimeError, match="dataset tree contains symlink"):
        reject_symlinks(dataset_root)

    source = UPSTREAM_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(UPSTREAM_SOURCE))
    save_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_successful_episode"
    )
    assert any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "reject_dataset_tree_symlinks"
        for child in ast.walk(save_function)
    )


def test_child_rejects_dataset_root_symlink_before_persistence(tmp_path: Path) -> None:
    outside = tmp_path / "outside-root"
    outside.mkdir()
    dataset_root = tmp_path / "episodes"
    dataset_root.symlink_to(outside, target_is_directory=True)
    namespace: dict[str, object] = {"os": os}
    reject_symlinks = _extract_function("reject_dataset_tree_symlinks", namespace)

    with pytest.raises(RuntimeError, match="dataset root is symlink"):
        reject_symlinks(dataset_root)


def test_finally_explicitly_finalizes_dataset() -> None:
    module = ast.parse(
        UPSTREAM_SOURCE.read_text(encoding="utf-8"),
        filename=str(UPSTREAM_SOURCE),
    )
    finalizers = [node.finalbody for node in module.body if isinstance(node, ast.Try)]

    assert any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "finalize_dataset"
        for finalizer in finalizers
        for statement in finalizer
        for child in ast.walk(statement)
    )


def test_stop_recording_flushes_once_and_clears_active_buffer() -> None:
    frames = [_frame(0.0), _frame(0.2)]
    calls: list[tuple[list[dict[str, object]], bool]] = []

    def save_successful_episode(
        captured: list[dict[str, object]],
        succeeded: bool,
    ) -> bool:
        calls.append((captured, succeeded))
        return True

    namespace: dict[str, object] = {
        "is_recording": True,
        "record_buffer": frames.copy(),
        "is_success": True,
        "save_successful_episode": save_successful_episode,
        "logger": SimpleNamespace(info=_ignore),
    }
    stop = _extract_function("stop_recording", namespace)

    assert stop() is True
    assert calls == [(frames, True)]
    assert namespace["is_recording"] is False
    assert namespace["record_buffer"] == []


def test_each_recording_requires_a_new_episode_local_success() -> None:
    calls: list[tuple[list[object], bool]] = []

    def save_successful_episode(captured: list[object], succeeded: bool) -> bool:
        calls.append((captured, succeeded))
        return succeeded

    namespace: dict[str, object] = {
        "is_recording": False,
        "record_buffer": [],
        "is_success": False,
        "save_successful_episode": save_successful_episode,
        "logger": SimpleNamespace(info=_ignore),
    }
    stop = _extract_function("stop_recording", namespace)
    namespace["stop_recording"] = stop
    toggle = _extract_function("record_toggle", namespace)

    toggle()
    namespace["record_buffer"] = ["successful-take-frame"]
    namespace["is_success"] = True
    toggle()

    toggle()
    assert namespace["is_success"] is False
    namespace["record_buffer"] = ["unsuccessful-take-frame"]
    toggle()

    assert calls == [
        (["successful-take-frame"], True),
        (["unsuccessful-take-frame"], False),
    ]


def test_solved_scene_requires_new_unsolved_to_solved_transition() -> None:
    calls: list[tuple[list[object], bool]] = []

    def save_successful_episode(captured: list[object], succeeded: bool) -> bool:
        calls.append((captured, succeeded))
        return succeeded

    namespace: dict[str, object] = {
        "is_recording": False,
        "record_buffer": [],
        "is_success": False,
        "episode_seen_unsolved": False,
        "save_successful_episode": save_successful_episode,
        "logger": SimpleNamespace(info=_ignore),
    }
    stop = _extract_function("stop_recording", namespace)
    namespace["stop_recording"] = stop
    toggle = _extract_function("record_toggle", namespace)
    update_success = _extract_function("update_episode_success", namespace)

    toggle()
    assert update_success(True) is False
    namespace["record_buffer"] = ["no-op-post-success-frame"]
    toggle()

    toggle()
    assert update_success(False) is False
    assert update_success(True) is True
    namespace["record_buffer"] = ["new-transition-frame"]
    toggle()

    assert calls == [
        (["no-op-post-success-frame"], False),
        (["new-transition-frame"], True),
    ]


def test_finally_flushes_active_recording_for_viewer_close_or_exception() -> None:
    module = ast.parse(
        UPSTREAM_SOURCE.read_text(encoding="utf-8"),
        filename=str(UPSTREAM_SOURCE),
    )
    finalizers = [node.finalbody for node in module.body if isinstance(node, ast.Try)]

    assert any(
        isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "is_recording"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "stop_recording"
            for child in ast.walk(statement)
        )
        for finalizer in finalizers
        for root in finalizer
        for statement in ast.walk(root)
    )


def test_upstream_opens_only_capable_index_zero_f710() -> None:
    wrong = FakeJoystick("Xbox Controller", 6, 11)

    def open_wrong(index: int) -> FakeJoystick:
        assert index == 0
        return wrong

    namespace: dict[str, object] = {
        "pygame": SimpleNamespace(
            joystick=SimpleNamespace(get_count=lambda: 2, Joystick=open_wrong)
        ),
        "F710_NAME": "Logitech Gamepad F710",
        "F710_MIN_AXES": 4,
        "F710_MIN_BUTTONS": 8,
    }
    open_f710 = _extract_function("open_f710_at_index_zero", namespace)

    with pytest.raises(RuntimeError, match="index 0 is not a capable Logitech Gamepad F710"):
        open_f710()

    capable = FakeJoystick("Logitech Gamepad F710", 6, 11)

    def open_capable(index: int) -> FakeJoystick:
        assert index == 0
        return capable

    namespace["pygame"] = SimpleNamespace(
        joystick=SimpleNamespace(get_count=lambda: 1, Joystick=open_capable)
    )

    assert open_f710() is capable
    assert capable.initialized is True


def test_active_pose_helper_has_no_direct_torchvision_import() -> None:
    module = ast.parse(UPSTREAM_HELPER.read_text(encoding="utf-8"), filename=str(UPSTREAM_HELPER))
    imported = {
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "torchvision.transforms" not in imported
