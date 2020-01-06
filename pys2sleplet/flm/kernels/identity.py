from typing import List, Optional

import numpy as np

from ..functions import Functions


class Identity(Functions):
    def __init__(self, L: int, args: List[int] = None):
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        raise NotImplementedError

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.ones((L * L)) + 1j * np.zeros((L * L))
        return flm

    def _create_name(self) -> str:
        name = "identity"
        return name
