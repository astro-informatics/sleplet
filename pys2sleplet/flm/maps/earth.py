import os

import numpy as np
import pyssht as ssht
from scipy import io as sio

from pys2sleplet.flm.functions import Functions


class Earth(Functions):
    def __init__(self, sig=3.0):
        super().__init__("earth", reality=True)
        self.sig = sig

    @staticmethod
    def load_flm():
        matfile = os.path.join(
            __location__, "data", "EGM2008_Topography_flms_L2190.mat"
        )
        mat_contents = sio.loadmat(matfile)
        flm = np.ascontiguousarray(mat_contents["flm"][:, 0])
        return flm

    def create_flm(self):
        # load in data
        flm = self.load_flm()

        # fill in negative m components so as to
        # avoid confusion with zero values
        for ell in range(1, self.L):
            for m in range(1, ell + 1):
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])

        # don't take the full L
        # invert dataset as Earth backwards
        flm = np.conj(flm[: self.L * self.L])
        return flm
