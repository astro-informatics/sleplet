from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht
from scipy import io as sio

from ..functions import Functions


class WMAP(Functions):
    def __init__(self, L: int, args: List[int] = None) -> None:
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        pass

    def _create_flm(self, L: int) -> np.ndarray:
        # load in data
        cl = self.load_cl()

        # same random seed
        np.random.seed(0)

        # Simulate CMB in harmonic space.
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(2, L):
            cl[ell - 1] = cl[ell - 1] * 2 * np.pi / (ell * (ell + 1))
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                if m == 0:
                    flm[ind] = np.sqrt(cl[ell - 1]) * np.random.randn()
                else:
                    flm[ind] = (
                        np.sqrt(cl[ell - 1] / 2) * np.random.randn()
                        + 1j * np.sqrt(cl[ell - 1] / 2) * np.random.randn()
                    )
        return flm

    def _create_name(self) -> str:
        name = "wmap"
        return name

    def _create_annotations(self) -> List[Dict]:
        pass

    @staticmethod
    def load_cl(file_ending="_lcdm_pl_model_wmap7baoh0.mat"):
        """
        pick coefficients from file options are:
        * _lcdm_pl_model_yr1_v1.mat
        * _tt_spectrum_7yr_v4p1.mat
        * _lcdm_pl_model_wmap7baoh0.mat
        """
        matfile = str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "maps"
            / "wmap"
            / f"wmap{file_ending}"
        )
        mat_contents = sio.loadmat(matfile)
        cl = np.ascontiguousarray(mat_contents["cl"][:, 0])
        return cl
