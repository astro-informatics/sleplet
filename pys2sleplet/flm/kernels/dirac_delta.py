import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions


class DiracDelta(Functions):
    def __init__(self, L: int):
        name = "dirac_delta"
        reality = True
        super().__init__(name, L, reality)

    def create_flm(self) -> np.ndarray:
        flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return flm
