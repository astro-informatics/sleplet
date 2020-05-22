from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def boost_flm_resolution(flm: np.ndarray, L: int, resolution: int) -> np.ndarray:
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
        flm = boost_flm_resolution(flm, L, resolution)
        bandlimit = resolution
    else:
        bandlimit = L

    f = ssht.inverse(flm, bandlimit, Reality=reality, Method=SAMPLING_SCHEME)
    return f
