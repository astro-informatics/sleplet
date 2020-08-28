from typing import Callable

import numpy as np
import pyssht as ssht


def create_spherical_harmonic(L: int, ind: int) -> np.ndarray:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    flm = np.zeros(L ** 2, dtype=np.complex128)
    flm[ind] = 1
    return flm


def _boost_flm_resolution(flm: np.ndarray, L: int, resolution: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    boost = resolution ** 2 - L ** 2
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: np.ndarray,
    L: int,
    resolution: int,
    method: str = "MW",
    reality: bool = False,
    spin: int = 0,
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    flm = _boost_flm_resolution(flm, L, resolution)
    return ssht.inverse(flm, resolution, Method=method, Reality=reality, Spin=spin)


def ensure_f_bandlimited(
    grid_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    L: int,
    reality: bool,
    spin: int,
) -> np.ndarray:
    """
    if the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it
    """
    thetas, phis = ssht.sample_positions(L, Grid=True)
    f = grid_fun(thetas, phis)
    return ssht.forward(f, L, Reality=reality, Spin=spin)


def create_emm_vector(L: int) -> np.ndarray:
    """
    create vector of m values for a given L
    """
    emm = np.zeros(2 * L * 2 * L)
    k = 0

    for l in range(2 * L):
        M = 2 * l + 1
        emm[k : k + M] = np.arange(-l, l + 1)
        k += M
    return emm
