# noqa: D100
import numpy as np
import numpy.typing as npt
import scipy.io as sio

import s2fft

import sleplet._data.setup_pooch
import sleplet._smoothing


def create_flm(L: int, *, smoothing: int | None = None) -> npt.NDArray[np.complex_]:
    """Creates the flm for the whole Earth."""
    # load in data
    flm_full = _load_flm()

    # don't take the full L, and convert to 2D
    flm_2d = s2fft.samples.flm_1d_to_2d(flm_full[: L**2], L)

    # fill in negative m components so as to avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            flm_2d[ell, L - 1 - m] = (-1) ** m * flm_2d[
                ell,
                L - 1 + m,
            ].conj()

    # invert dataset as Earth backwards
    flm = s2fft.samples.flm_2d_to_1d(flm_2d.conj(), L)

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
