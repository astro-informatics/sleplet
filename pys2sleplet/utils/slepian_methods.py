from pathlib import Path

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()


def choose_slepian_method() -> SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    phi_min = np.deg2rad(config.PHI_MIN)
    phi_max = np.deg2rad(config.PHI_MAX)
    theta_min = np.deg2rad(config.THETA_MIN)
    theta_max = np.deg2rad(config.THETA_MAX)

    if is_polar_cap(phi_min, phi_max, theta_min, theta_max):
        logger.info("polar cap region detected")
        slepian = SlepianPolarCap(config.L, theta_max, order=config.ORDER)

    elif is_limited_lat_lon(phi_min, phi_max, theta_min, theta_max):
        logger.info("limited latitude longitude region detected")
        slepian = SlepianLimitLatLong(config.L, theta_min, theta_max, phi_min, phi_max)

    elif config.SLEPIAN_MASK:
        logger.info("mask specified in file detected")
        mask = _load_mask(config.SLEPIAN_MASK)
        slepian = SlepianArbitrary(config.L, mask, config.SLEPIAN_MASK)

    else:
        raise AttributeError(
            "need to specify either a polar cap, a limited latitude "
            "longitude region, or a file with a mask"
        )

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
