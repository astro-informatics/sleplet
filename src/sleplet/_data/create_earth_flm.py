import numpy as np
import numpy.typing as npt
import scipy.io as sio

import pyssht as ssht

import sleplet._data.setup_pooch
import sleplet._smoothing


def create_flm(L: int, *, smoothing: int | None = None) -> npt.NDArray[np.complex_]:
    """Create the flm for the whole Earth."""
    # load in data
    flm = _load_flm()

    # fill in negative m components so as to avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm_pm = flm[ind_pm]
            flm[ind_nm] = (-1) ** m * flm_pm.conj()

    # don't take the full L, invert dataset as Earth backwards
    flm = flm[: L**2].conj()

    if isinstance(smoothing, int):
        flm = sleplet._smoothing.apply_gaussian_smoothing(flm, L, smoothing)
    return flm


def _load_flm() -> npt.NDArray[np.complex_]:
    """Load coefficients from file."""
    mat_contents = sio.loadmat(
        sleplet._data.setup_pooch.find_on_pooch_then_local(
            "EGM2008_Topography_flms_L2190.mat",
        ),
    )
    return np.ascontiguousarray(mat_contents["flm"][:, 0])
