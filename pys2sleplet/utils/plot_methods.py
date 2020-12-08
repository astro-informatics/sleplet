from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyssht as ssht
from matplotlib import colors
from matplotlib import pyplot as plt

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import find_non_zero_wavelet_coefficients

_file_location = Path(__file__).resolve()
_amplitude_path = _file_location.parents[1] / "data" / "slepian" / "amplitudes"


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
    logger.info(f"grid point: (alpha, beta)=({alpha:e}, {beta:e})")
    return alpha, beta


def save_plot(path: Path, name: str) -> None:
    """
    helper method to save plots
    """
    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = path / file_type / f"{name}.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def find_max_amplitude(
    L: int,
    coefficients: np.ndarray,
    slepian: Optional[SlepianFunctions] = None,
    coefficient_type: str = "",
) -> Dict[str, float]:
    """
    computes the maximum value for the given array in
    pixel space, for the real, imaginary & complex parts
    """
    logger.info("starting: find maximum amplitude values")
    if isinstance(slepian, SlepianFunctions):
        amplitude_loc = (
            _amplitude_path
            / f"{slepian.region.name_ending}_L{L}_N{slepian.N}_{coefficient_type}.npy"
        )
        if amplitude_loc.exists():
            logger.info("amplitude binaries found")
            amplitudes = np.load(amplitude_loc, allow_pickle=True).item()
        else:
            within_shannon_coefficients = find_non_zero_wavelet_coefficients(
                coefficients
            )
            amplitudes = _create_max_amplitues_dict(
                np.apply_along_axis(
                    lambda c: slepian_inverse(L, c, slepian),
                    1,
                    within_shannon_coefficients,
                )
            )
            if settings.SAVE_MATRICES:
                np.save(amplitude_loc, amplitudes)
    else:
        amplitudes = _create_max_amplitues_dict(
            np.apply_along_axis(
                lambda c: ssht.inverse(c, L, Method=SAMPLING_SCHEME), 1, coefficients
            )
        )
    logger.info("finished: find maximum amplitude values")
    return amplitudes


def _create_max_amplitues_dict(fields: np.ndarray) -> Dict[str, float]:
    """
    creates the dictionary of the max value for each plot type
    """
    return dict(
        abs=np.abs(fields).max(),
        imag=fields.imag.max(),
        real=fields.real.max(),
        sum=(fields.real + fields.imag).max(),
    )
