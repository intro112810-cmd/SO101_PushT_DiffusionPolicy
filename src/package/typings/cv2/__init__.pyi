"""Minimal type stub for the subset of OpenCV used by the LiveViewer overlay.

Only the symbols exercised at display time are declared here; keep this in
sync with `collection/viewer.py`. The package intentionally does not vendor
full cv2 typing — cv2 is a display-time-only dependency.
"""

from typing import TypeAlias

from numpy.typing import NDArray
import numpy as np

_MAT: TypeAlias = NDArray[np.uint8]

# Marker/font constants used by the overlay (cv2 module attributes).
MARKER_CROSS: int
FONT_HERSHEY_SIMPLEX: int
LINE_8: int
INTER_AREA: int
INTER_LINEAR: int
SOLVEPNP_ITERATIVE: int

def resize(
    src: NDArray[np.uint8],
    dsize: tuple[int, int],
    interpolation: int = ...,
) -> NDArray[np.uint8]: ...
def circle(
    img: _MAT,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int = -1,
    lineType: int | None = None,
    shift: int = 0,
) -> _MAT: ...
def drawMarker(
    img: _MAT,
    position: tuple[int, int],
    color: tuple[int, int, int],
    markerType: int = ...,
    markerSize: int = 20,
    thickness: int = 1,
    line_type: int = ...,
) -> _MAT: ...
def putText(
    img: _MAT,
    text: str,
    org: tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    lineType: int = ...,
    bottomLeftOrigin: bool = False,
) -> _MAT: ...

IMREAD_COLOR: int
ROTATE_90_CLOCKWISE: int
ROTATE_90_COUNTERCLOCKWISE: int

def imread(path: str, flags: int = ...) -> NDArray[np.uint8] | None: ...
def rotate(src: NDArray[np.uint8], rotateCode: int) -> NDArray[np.uint8]: ...
def solvePnP(
    object_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    flags: int = ...,
) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]: ...
def projectPoints(
    object_points: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
