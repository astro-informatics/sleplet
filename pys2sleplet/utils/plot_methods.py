from typing import Callable, List, Tuple

import numpy as np
import pyssht as ssht
from matplotlib import colors


def calc_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    if L == 1:
        exponent = 6
    elif L < 4:
        exponent = 5
    elif L < 8:
        exponent = 4
    elif L < 128:
        exponent = 3
    elif L < 512:
        exponent = 2
    elif L < 1024:
        exponent = 1
    else:
        exponent = 0
    return L * 2 ** exponent


def calc_samples(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    if L == 1:
        samples = 1801
    elif L < 4:
        samples = 901
    elif L < 8:
        samples = 451
    elif L < 16:
        samples = 226
    elif L < 32:
        samples = 113
    elif L < 64:
        samples = 57
    elif L < 128:
        samples = 29
    elif L < 256:
        samples = 15
    elif L < 512:
        samples = 8
    elif L < 1024:
        samples = 4
    elif L < 2048:
        samples = 2
    else:
        samples = 1
    return samples


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
):
    thetas, phis = ssht.sample_positions(L, Grid=True, Method="MWSS")
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, L, Reality=reality, Method="MWSS")
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
    thetas, phis = ssht.sample_positions(L, Method="MWSS")
    pix_j = (np.abs(phis - alpha_pi_fraction * np.pi)).argmin()
    pix_i = (np.abs(thetas - beta_pi_fraction * np.pi)).argmin()
    alpha, beta = phis[pix_j], thetas[pix_i]
    return alpha, beta
