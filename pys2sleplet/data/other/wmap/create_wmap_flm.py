from pathlib import Path

import numpy as np
import pyssht as ssht
from numpy.random import default_rng
from scipy import io as sio

from pys2sleplet.utils.vars import RANDOM_SEED

_file_location = Path(__file__).resolve()


def create_flm(L: int) -> np.ndarray:
    """
    creates the flm for the whole CMB
    """
    # load in data
    cl = _load_cl()

    # same random seed
    rng = default_rng(RANDOM_SEED)

    # Simulate CMB in harmonic space.
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(2, L):
        cl_val = cl[ell - 1]
        cl_val *= 2 * np.pi / (ell * (ell + 1))
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            if m == 0:
                flm[ind] = np.sqrt(cl_val) * rng.standard_normal()
            else:
                flm[ind] = (
                    np.sqrt(cl_val / 2) * rng.standard_normal()
                    + 1j * np.sqrt(cl_val / 2) * rng.standard_normal()
                )
    return flm


def _load_cl(file_ending: str = "_lcdm_pl_model_wmap7baoh0") -> np.ndarray:
    """
    pick coefficients from file options are:
    * _lcdm_pl_model_yr1_v1.mat
    * _tt_spectrum_7yr_v4p1.mat
    * _lcdm_pl_model_wmap7baoh0.mat
    """
    matfile = str(_file_location.parent / f"wmap{file_ending}.mat")
    mat_contents = sio.loadmat(matfile)
    return np.ascontiguousarray(mat_contents["cl"][:, 0])
