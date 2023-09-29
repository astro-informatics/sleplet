# noqa: D100
import numpy as np
import numpy.typing as npt
import scipy.io as sio

import s2fft

import sleplet._data.setup_pooch
import sleplet._smoothing


def create_flm(
    L: int,
    *,
    smoothing: int | None = None,
) -> npt.NDArray[np.complex_]:
    """Creates the flm for the whole Earth."""
    # load in data
    flm, L_full = _load_flm()

    # fill in negative m components so as to
    # avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            flm_pm = flm[ell, L + m - 1]
            flm[ell, L - m - 1] = (-1) ** m * flm_pm.conj()

    # don't take the full L
    # invert dataset as Earth backwards
    flm = flm[:L, L_full - L - 1 : L_full + L - 2].conj()

    if isinstance(smoothing, int):
        flm = sleplet._smoothing.apply_gaussian_smoothing(flm, L, smoothing)
    return flm


def _load_flm() -> tuple[npt.NDArray[np.complex_], int]:
    """Load coefficients from file."""
    mat_contents = sio.loadmat(
        sleplet._data.setup_pooch.find_on_pooch_then_local(
            "EGM2008_Topography_flms_L2190.mat",
        ),
    )
    L = int(mat_contents["L"])
    flm = s2fft.sampling.s2_samples.flm_1d_to_2d(
        np.ascontiguousarray(mat_contents["flm"][:, 0]),
        L,
    )
    return flm, L
