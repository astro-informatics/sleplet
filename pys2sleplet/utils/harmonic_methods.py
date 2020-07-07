from typing import Callable

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def create_spherical_harmonic(L: int, ind: int) -> np.ndarray:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    flm = np.zeros(L * L, dtype=complex)
    flm[ind] = 1
    return flm


def _boost_flm_resolution(flm: np.ndarray, L: int, resolution: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    boost = resolution * resolution - L * L
    flm_boost = np.pad(flm, (0, boost), "constant")
    return flm_boost


def invert_flm_boosted(
    flm: np.ndarray, L: int, resolution: int, reality: bool = False
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    flm = _boost_flm_resolution(flm, L, resolution)
    f = ssht.inverse(flm, resolution, Reality=reality, Method=SAMPLING_SCHEME)
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


def create_emm_vector(L: int) -> np.ndarray:
    """
    create vector of m values for a given L
    """
    emm = np.zeros(2 * L * 2 * L)
    k = 0

    for l in range(2 * L):
        M = 2 * l + 1
        emm[k : k + M] = np.arange(-l, l + 1)
        k = k + M
    return emm
