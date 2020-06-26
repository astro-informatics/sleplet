from typing import List, Tuple

import numpy as np
import pyssht as ssht
from matplotlib import colors

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def calc_plot_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    for log_bandlimit, exponent in res_dict.items():
        if L < 2 ** log_bandlimit:
            return L * 2 ** exponent

    # otherwise just use the bandlimit
    return L


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
