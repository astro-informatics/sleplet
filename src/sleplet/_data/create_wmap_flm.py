import numpy as np
import numpy.typing as npt
import scipy.io as sio

import pyssht as ssht

import sleplet._data.setup_pooch
import sleplet._vars


def create_flm(L: int) -> npt.NDArray[np.complex_]:
    """Create the flm for the whole CMB."""
    # load in data
    cl = _load_cl()

    # same random seed
    rng = np.random.default_rng(sleplet._vars.RANDOM_SEED)

    # Simulate CMB in harmonic space.
    flm = np.zeros(L**2, dtype=np.complex_)
    for ell in range(2, L):
        sigma = np.sqrt(2 * np.pi / (ell * (ell + 1)) * cl[ell - 2])
        ind = ssht.elm2ind(ell, 0)
        flm[ind] = sigma * rng.standard_normal()
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm[ind_pm] = (
                sigma
                / np.sqrt(2)
                * (rng.standard_normal() + 1j * rng.standard_normal())
            )
            flm[ind_nm] = (-1) ** m * flm[ind_pm].conj()
    return flm


def _load_cl(
    *,
    file_ending: str = "_lcdm_pl_model_wmap7baoh0",
) -> npt.NDArray[np.float_]:
    """
    Pick coefficients from file options are:
    * _lcdm_pl_model_yr1_v1.mat
    * _tt_spectrum_7yr_v4p1.mat
    * _lcdm_pl_model_wmap7baoh0.mat.
    """
    mat_contents = sio.loadmat(
        sleplet._data.setup_pooch.find_on_pooch_then_local(f"wmap{file_ending}.mat"),
    )
    return np.ascontiguousarray(mat_contents["cl"][:, 0])
