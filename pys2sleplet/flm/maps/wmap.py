from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyssht as ssht
from scipy import io as sio

from pys2sleplet.flm.functions import Functions

_file_location = Path(__file__).resolve()


@dataclass
class Wmap(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self) -> np.ndarray:
        # load in data
        cl = self._load_cl()

        # same random seed
        np.random.seed(0)

        # Simulate CMB in harmonic space.
        flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(2, self.L):
            cl_val = cl[ell - 1]
            cl_val *= 2 * np.pi / (ell * (ell + 1))
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                if m == 0:
                    flm[ind] = np.sqrt(cl_val) * np.random.randn()
                else:
                    flm[ind] = (
                        np.sqrt(cl_val / 2) * np.random.randn()
                        + 1j * np.sqrt(cl_val / 2) * np.random.randn()
                    )
        return flm

    def _create_name(self) -> str:
        name = "wmap"
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            raise AttributeError(f"Does not support extra arguments")

    @staticmethod
    def _load_cl(file_ending: str = "_lcdm_pl_model_wmap7baoh0") -> np.ndarray:
        """
        pick coefficients from file options are:
        * _lcdm_pl_model_yr1_v1.mat
        * _tt_spectrum_7yr_v4p1.mat
        * _lcdm_pl_model_wmap7baoh0.mat
        """
        filename = f"wmap{file_ending}.mat"
        matfile = str(_file_location.parents[2] / "data" / "maps" / "wmap" / filename)
        mat_contents = sio.loadmat(matfile)
        cl = np.ascontiguousarray(mat_contents["cl"][:, 0])
        return cl
