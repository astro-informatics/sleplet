from typing import Callable, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def _boost_flm_resolution(flm: np.ndarray, L: int, resolution: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    boost = resolution * resolution - L * L
    flm_boost = np.pad(flm, (0, boost), "constant")
    return flm_boost


def invert_flm(
    flm: np.ndarray, L: int, reality: bool = False, resolution: Optional[int] = None
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    if resolution is not None:
        flm = _boost_flm_resolution(flm, L, resolution)
        bandlimit = resolution
    else:
        bandlimit = L

    f = ssht.inverse(flm, bandlimit, Reality=reality, Method=SAMPLING_SCHEME)
    return f


def ensure_f_bandlimited(
    grid_fun: Callable[[np.ndarray, np.ndarray], np.ndarray], L: int, reality: bool
) -> np.ndarray:
    """
    if the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it
    """
    theta_grid, phi_grid = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    f = grid_fun(theta_grid, phi_grid)
    flm = ssht.forward(f, L, Reality=reality, Method=SAMPLING_SCHEME)
    return flm
