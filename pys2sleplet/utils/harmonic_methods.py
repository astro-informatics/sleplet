from typing import Callable  # TODO: import from collections.abc

import numpy as np
import pyssht as ssht
from numpy.random import Generator

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def create_spherical_harmonic(L: int, ind: int) -> np.ndarray:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    flm = np.zeros(L ** 2, dtype=np.complex_)
    flm[ind] = 1
    return flm


def boost_coefficient_resolution(flm: np.ndarray, boost: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: np.ndarray, L: int, resolution: int, reality: bool = False, spin: int = 0
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    boost = resolution ** 2 - L ** 2
    flm = boost_coefficient_resolution(flm, boost)
    return ssht.inverse(
        flm, resolution, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME
    )


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
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    f = grid_fun(thetas, phis)
    return ssht.forward(f, L, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME)


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


def compute_random_signal(L: int, rng: Generator, var_signal: float) -> np.ndarray:
    """
    generates a normally distributed random signal of a
    complex signal with mean 0 and variance 1
    """
    return np.sqrt(var_signal / 2) * (
        rng.standard_normal(L ** 2) + 1j * rng.standard_normal(L ** 2)
    )
