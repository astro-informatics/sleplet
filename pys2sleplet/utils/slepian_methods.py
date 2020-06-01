from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config
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
    theta_grid, phi_grid = ssht.sample_position(L, Grid=True, Method=SAMPLING_SCHEME)

    if region.region_type == "polar":
        logger.info("creating polar cap mask")
        mask = theta_grid <= region.theta_max

    elif region.region_type == "lim_lat_lon":
        logger.info("creating limited latitude longitude mask")
        mask = (
            (theta_grid >= region.theta_min)
            & (theta_grid <= region.theta_max)
            & (phi_grid >= region.phi_min)
            & (phi_grid <= region.phi_max)
        )

    elif region.region_type == "arbitrary":
        logger.info("loading and checking shape of provided mask")
        mask = _load_mask(region.mask_name)
        assert mask.shape == theta_grid.shape, (
            f"mask shape {mask.shape} does not match the provided "
            f"L={L}, the shape should be {theta_grid.shape}"
        )

    return mask


def choose_slepian_method() -> SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    region = Region(
        phi_min=np.deg2rad(config.PHI_MIN),
        phi_max=np.deg2rad(config.PHI_MAX),
        theta_min=np.deg2rad(config.THETA_MIN),
        theta_max=np.deg2rad(config.THETA_MAX),
        mask_name=config.SLEPIAN_MASK,
    )

    if region.region_type == "polar":
        logger.info("polar cap region detected")
        slepian = SlepianPolarCap(config.L, region.theta_max, order=config.ORDER)

    elif region.region_type == "lim_lat_lon":
        logger.info("limited latitude longitude region detected")
        slepian = SlepianLimitLatLon(
            config.L, region.theta_min, region.theta_max, region.phi_min, region.phi_max
        )

    elif region.region_type == "arbitrary":
        logger.info("mask specified in file detected")
        slepian = SlepianArbitrary(config.L, region.mask_name)

    return slepian


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
