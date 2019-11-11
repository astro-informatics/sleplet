import numpy as np
import pyssht as ssht

from ..functions import Functions


class DiracDelta(Functions):
    def __init__(self):
        super().__init__("dirac_delta", L, reality=True)

    @Functions.flm.setter
    def flm(self):
        self._flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, m=0)
            self._flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
