from typing import Callable, List, Tuple

import numpy as np
import pyssht as ssht
from matplotlib import colors

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def calc_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    for log_bandlimit, exponent in res_dict.items():
        if L < 2 ** log_bandlimit:
            return L * 2 ** exponent

    # above L = 1024 just use the bandlimit
    return L


def calc_samples(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    sample_dict = {
        1: 1801,
        2: 901,
        3: 451,
        4: 226,
        5: 113,
        6: 57,
        7: 29,
        8: 15,
        9: 8,
        10: 4,
        11: 2,
    }

    for log_bandlimit, samples in sample_dict.items():
        if L < 2 ** log_bandlimit:
            return samples

    # above L = 2048 just use 1 sample
    return 1


def convert_colourscale(cmap: colors, pl_entries: int = 255) -> List[Tuple[float, str]]:
    """
    converts cmocean colourscale to a plotly colourscale
    """
    h = 1 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append((k * h, f"rgb{(C[0], C[1], C[2])}"))

    return pl_colorscale


def ensure_f_bandlimited(
    grid_fun: Callable[[np.ndarray, np.ndarray], np.ndarray], L: int, reality: bool
) -> np.ndarray:
    """
    if the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it
    """
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, L, Reality=reality, Method=SAMPLING_SCHEME)
    return flm


def calc_nearest_grid_point(
    L: int, alpha_pi_fraction: float, beta_pi_fraction: float
) -> Tuple[float, float]:
    """
    calculate nearest index of alpha/beta for translation
    this is due to calculating omega' through the pixel
    values - the translation needs to be at the same position
    as the rotation such that the difference error is small
    """
    thetas, phis = ssht.sample_positions(L, Method=SAMPLING_SCHEME)
    pix_j = np.abs(phis - alpha_pi_fraction * np.pi).argmin()
    pix_i = np.abs(thetas - beta_pi_fraction * np.pi).argmin()
    alpha, beta = phis[pix_j], thetas[pix_i]
    return alpha, beta
