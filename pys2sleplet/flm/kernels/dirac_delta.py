from typing import List

import numpy as np
import pyssht as ssht

from ..functions import Functions


class DiracDelta(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = True
        super().__init__(L)

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return flm

    def _create_name(self) -> str:
        name = "dirac_delta"
        return name
