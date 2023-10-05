# noqa: D100
import numpy as np
import numpy.typing as npt
import scipy.io as sio

import s2fft

import sleplet._data.setup_pooch
import sleplet._vars


def create_flm(L: int) -> npt.NDArray[np.complex_]:
    """Creates the flm for the whole CMB."""
    # load in data
    cl = _load_cl()

    # same random seed
    rng = np.random.default_rng(sleplet._vars.RANDOM_SEED)

    # Simulate CMB in harmonic space.
    flm = np.zeros(s2fft.samples.flm_shape(L), dtype=np.complex_)
    for ell in range(2, L):
        sigma = np.sqrt(2 * np.pi / (ell * (ell + 1)) * cl[ell - 2])
        flm[ell, L - 1] = sigma * rng.standard_normal()
        for m in range(1, ell + 1):
            ind_pm = L - 1 + m
            ind_nm = L - 1 - m
            flm[ell, ind_pm] = (
                sigma
                / np.sqrt(2)
                * (rng.standard_normal() + 1j * rng.standard_normal())
            )
            flm[ell, ind_nm] = (-1) ** m * flm[ell, ind_pm].conj()
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
