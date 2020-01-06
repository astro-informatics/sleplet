from pathlib import Path
from typing import List, Optional

import numpy as np
import pyssht as ssht
from scipy import io as sio

from ..functions import Functions


class Earth(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        raise NotImplementedError

    def _create_flm(self, L: int) -> np.ndarray:
        # load in data
        flm = self.load_flm()

        # fill in negative m components so as to
        # avoid confusion with zero values
        for ell in range(1, L):
            for m in range(1, ell + 1):
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])

        # don't take the full L
        # invert dataset as Earth backwards
        flm = np.conj(flm[: L * L])
        return flm

    def _create_name(self) -> str:
        name = "earth"
        return name

    @staticmethod
    def load_flm():
        matfile = str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "maps"
            / "earth"
            / "EGM2008_Topography_flms_L2190.mat"
        )
        mat_contents = sio.loadmat(matfile)
        flm = np.ascontiguousarray(mat_contents["flm"][:, 0])
        return flm
