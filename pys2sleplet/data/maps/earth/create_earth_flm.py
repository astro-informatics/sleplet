from pathlib import Path

import numpy as np
import pyssht as ssht
from scipy import io as sio

_file_location = Path(__file__).resolve()


def create_flm(L: int) -> np.ndarray:
    """
    creates the flm for the whole Earth
    """
    # load in data
    flm = _load_flm()

    # fill in negative m components so as to
    # avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm_pm = flm[ind_pm]
            flm[ind_nm] = (-1) ** m * flm_pm.conj()

    # don't take the full L
    # invert dataset as Earth backwards
    flm = flm[: L ** 2].conj()
    return flm


def _load_flm() -> np.ndarray:
    """
    load coefficients from file
    """
    matfile = str(_file_location.parent / "EGM2008_Topography_flms_L2190.mat")
    mat_contents = sio.loadmat(matfile)
    return np.ascontiguousarray(mat_contents["flm"][:, 0])
