from pathlib import Path

import numpy as np
from numpy.typing import NDArray

def imwrite(uri: str | Path, image: NDArray[np.generic], *, fps: int) -> None: ...
