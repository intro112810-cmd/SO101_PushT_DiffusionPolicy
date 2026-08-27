from collections.abc import Iterator
from pathlib import Path
from fractions import Fraction
import numpy as np
from numpy.typing import NDArray

class VideoStream:
    average_rate: Fraction | None
    width: int
    height: int

class _VideoStreams:
    video: list[VideoStream]

class VideoFrame:
    pts: int | None
    time_base: Fraction
    def to_ndarray(self, *, format: str) -> NDArray[np.uint8]: ...

class InputContainer:
    streams: _VideoStreams
    def decode(self, *, video: int) -> Iterator[VideoFrame]: ...
    def close(self) -> None: ...

def open(path: str | Path) -> InputContainer: ...
