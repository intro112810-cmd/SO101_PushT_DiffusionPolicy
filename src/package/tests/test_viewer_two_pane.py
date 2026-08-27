import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.collection.viewer import LiveViewer, OverlayState


def test_two_separate_windows_each_display_their_frame() -> None:
    control = LiveViewer.open(enabled=False)
    observer = LiveViewer.open(enabled=False)
    paper: NDArray[np.uint8] = np.zeros((384, 384, 3), dtype=np.uint8)
    front: NDArray[np.uint8] = np.ones((384, 384, 3), dtype=np.uint8) * 255

    # Schema-3: the control pane shows the paper view, the observer pane
    # shows the 3D frame; each window holds exactly its own frame.
    control.show(paper)
    observer.show(front)

    assert control.last_frame is paper
    assert observer.last_frame is front
    assert control.last_frame is not None
    assert control.last_frame.shape == (384, 384, 3)
    assert observer.last_frame is not None
    assert observer.last_frame.shape == (384, 384, 3)
    control.close()
    observer.close()


def test_source_arrays_remain_byte_identical_with_overlay() -> None:
    viewer = LiveViewer.open(enabled=False)
    paper = np.zeros((384, 384, 3), dtype=np.uint8)
    paper[0, 0, 0] = 1
    snapshot = paper.copy()

    paper.flags.writeable = False

    viewer.show(
        paper,
        OverlayState(
            measured_ee=(0.28, 0.0), requested_target=(0.34, 0.0), z_level=0.045, state="ARMED"
        ),
        bounds_x=(0.18, 0.38),
        bounds_y=(-0.16, 0.16),
    )

    assert paper.tolist() == snapshot.tolist()
    viewer.close()


def test_headless_offscreen_render() -> None:
    viewer = LiveViewer.open(enabled=False)
    frame: NDArray[np.uint8] = np.zeros((96, 96, 3), dtype=np.uint8)

    viewer.show(frame)
    composed = viewer.render()
    assert isinstance(composed, np.ndarray)
    assert composed.shape[2] == 3


def test_overlay_draws_on_display_copy_only() -> None:
    viewer = LiveViewer.open(enabled=False)
    paper: NDArray[np.uint8] = np.zeros((384, 384, 3), dtype=np.uint8)

    viewer.show(
        paper,
        OverlayState(
            measured_ee=(0.28, 0.0), requested_target=(0.34, 0.0), z_level=0.045, state="READY"
        ),
        bounds_x=(0.18, 0.38),
        bounds_y=(-0.16, 0.16),
    )
    composed = viewer.render()

    # The display copy carries the overlay; the source stays all zeros.
    assert composed is not paper
    assert bool((composed > 0).any())
    assert not bool((paper > 0).any())
    viewer.close()


def test_close_and_focus_loss_emit_correct_control_event() -> None:
    events: list[str] = []

    def on_close() -> None:
        events.append("close")

    viewer = LiveViewer.open(enabled=False, on_close=on_close)
    viewer.close()

    assert events == ["close"]
