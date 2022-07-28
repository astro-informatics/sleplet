from pathlib import Path

import numpy as np
import pyssht as ssht
from matplotlib import colors
from matplotlib import pyplot as plt

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import (
    EARTH_ALPHA,
    EARTH_BETA,
    EARTH_GAMMA,
    SAMPLING_SCHEME,
    UNSEEN,
)


def calc_plot_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    for log_bandlimit, exponent in res_dict.items():
        if L < 2**log_bandlimit:
            return L * 2**exponent

    # otherwise just use the bandlimit
    return L


def convert_colourscale(
    cmap: colors, *, pl_entries: int = 255
) -> list[tuple[float, str]]:
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
) -> tuple[float, float]:
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
    logger.info(f"grid point: (alpha, beta)=({alpha:e}, {beta:e})")
    return alpha, beta


def save_plot(path: Path, name: str) -> None:
    """
    helper method to save plots
    """
    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in {"png", "pdf"}:
            logger.info(f"saving {file_type}")
            filename = path / file_type / f"{name}.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def find_max_amplitude(
    function: Coefficients, *, plot_type: str = "real", upsample: bool = True
) -> dict[str, float]:
    """
    for a given set of coefficients it finds the largest absolute value for a
    given plot type such that plots can have the same scale as the input
    """
    # compute inverse transform
    if hasattr(function, "slepian"):
        field = slepian_inverse(function.coefficients, function.L, function.slepian)
    else:
        field = ssht.inverse(function.coefficients, function.L, Method=SAMPLING_SCHEME)

    # find resolution of final plot for boosting if necessary
    resolution = calc_plot_resolution(function.L) if upsample else function.L

    # boost field to match final plot
    boosted_field = boost_field(
        field, function.L, resolution, function.reality, function.spin, upsample
    )

    # find maximum absolute value for given plot type
    return np.abs(create_plot_type(boosted_field, plot_type)).max()


def create_plot_type(field: np.ndarray, plot_type: str) -> np.ndarray:
    """
    gets the given plot type of the field
    """
    logger.info(f"plotting type: '{plot_type}'")
    plot_dict = dict(
        abs=np.abs(field), imag=field.imag, real=field.real, sum=field.real + field.imag
    )
    return plot_dict[plot_type]


def set_outside_region_to_minimum(
    f_plot: np.ndarray, L: int, region: Region
) -> np.ndarray:
    """
    for the Slepian region set the outisde area to negative infinity
    hence it is clear we are only interested in the coloured region
    """
    # create mask of interest
    mask = create_mask_region(L, region)

    # adapt for closed plot
    _, n_phi = ssht.sample_shape(L, Method=SAMPLING_SCHEME)
    closed_mask = np.insert(mask, n_phi, mask[:, 0], axis=1)

    # set values outside mask to negative infinity
    return np.where(closed_mask, f_plot, UNSEEN)


def rotate_earth_to_south_america(earth_flm: np.ndarray, L: int) -> np.ndarray:
    """
    rotates the flms of the Earth to a view centered on South America
    """
    return ssht.rotate_flms(earth_flm, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, L)


def normalise_function(f: np.ndarray, normalise: bool) -> np.ndarray:
    """
    normalise function between 0 and 1 for visualisation
    """
    if not normalise:
        return f
    elif (f == 0).all():
        # if all 0, set to 0
        return f + 0.5
    elif np.allclose(f, f.max()):
        # if all non-zero, set to 1
        return f / f.max()
    else:
        # scale from [0, 1]
        return (f - f.min()) / f.ptp()


def boost_field(
    field: np.ndarray, L: int, resolution: int, reality: bool, spin: int, upsample: bool
) -> np.ndarray:
    """
    inverts and then boosts the field before plotting
    """
    if not upsample:
        return field
    flm = ssht.forward(field, L, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME)
    return invert_flm_boosted(flm, L, resolution, reality=reality, spin=spin)
