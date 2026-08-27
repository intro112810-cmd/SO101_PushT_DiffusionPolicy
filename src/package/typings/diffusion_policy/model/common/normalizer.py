from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


class SingleFieldLinearNormalizer:
    def normalize(self, value: Tensor | NDArray[np.generic]) -> Tensor: ...

    def unnormalize(self, value: Tensor | NDArray[np.generic]) -> Tensor: ...


class LinearNormalizer:
    def fit(
        self,
        data: Mapping[str, Tensor | NDArray[np.generic]],
        last_n_dims: int = 1,
        mode: str = "limits",
    ) -> None: ...

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        """Return the normalizer registered for one field."""

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        """Register one field normalizer."""

    def get_input_stats(self) -> dict[str, dict[str, Tensor]]: ...
