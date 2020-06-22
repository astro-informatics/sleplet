from pathlib import Path

import numexpr as ne
import numpy as np
import pyssht as ssht

from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import SAMPLING_SCHEME

_file_location = Path(__file__).resolve()


def create_mask_region(L: int, region: Region) -> np.ndarray:
    """
    creates a mask of a region of interested, the output will be based
    on the value of the provided L. The mask could be either:
    * polar cap - if theta_max provided
    * limited latitude longitude - if one of theta_min, theta_max,
                                   phi_min or phi_max is provided
    * arbitrary - just checks the shape of the input mask
    """
    theta_grid, phi_grid = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)

    if region.region_type == "polar":
        logger.info("creating polar cap mask")
        mask = ne.evaluate(f"theta_grid<={region.theta_max}")

    elif region.region_type == "lim_lat_lon":
        logger.info("creating limited latitude longitude mask")
        mask = ne.evaluate(
            f"(theta_grid>={region.theta_min})&(theta_grid<={region.theta_max})"
            f"&(phi_grid>={region.phi_min})&(phi_grid<={region.phi_max})"
        )

    elif region.region_type == "arbitrary":
        logger.info("loading and checking shape of provided mask")
        mask = _load_mask(region.mask_name)
        assert mask.shape == theta_grid.shape, (
            f"mask shape {mask.shape} does not match the provided "
            f"L={L}, the shape should be {theta_grid.shape}"
        )

    return mask


def _load_mask(mask_name: str) -> np.ndarray:
    """
    attempts to read the mask from the config file
    """
    location = (
        _file_location.parents[1]
        / "data"
        / "slepian"
        / "arbitrary"
        / "masks"
        / mask_name
    )
    try:
        mask = np.load(location)
    except FileNotFoundError:
        logger.error(f"can not find the file: '{mask_name}'")
        raise
    return mask


def ensure_masked_flm_bandlimited(
    flm: np.ndarray, L: int, region: Region, reality: bool
) -> np.ndarray:
    """
    ensures the multipole is bandlimited for a given region
    """
    field = ssht.inverse(flm, L, Reality=reality, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, region)
    field *= mask
    multipole = ssht.forward(field, L, Reality=reality, Method=SAMPLING_SCHEME)
    return multipole
