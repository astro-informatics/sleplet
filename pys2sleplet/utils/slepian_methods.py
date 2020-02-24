from pathlib import Path
from typing import Type

import numpy as np
from slepian.slepian_functions import SlepianFunctions
from slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from slepian.slepian_region.specific_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from slepian.slepian_region.specific_region.slepian_polar_cap import SlepianPolarCap
from utils.bool_methods import is_limited_lat_lon, is_polar_cap
from utils.inputs import config
from utils.logging import logger


def create_slepian(
    L: int, phi_min: float, phi_max: float, theta_min: float, theta_max: float
) -> Type[SlepianFunctions]:
    """
    initialise Slepian object depending on input
    """
    if is_polar_cap(phi_min, phi_max, theta_min, theta_max):
        logger.info("polar cap region detected")
        return SlepianPolarCap(L, theta_max, config.ORDER)

    elif is_limited_lat_lon(phi_min, phi_max, theta_min, theta_max):
        logger.info("limited latitude longitude region detected")
        return SlepianLimitLatLong(L, theta_min, theta_max, phi_min, phi_max)

    elif config.SLEPIAN_MASK:
        logger.info("no angles specified, looking for a file with mask")
        location = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "slepian"
            / "arbitrary"
            / "masks"
            / config.SLEPIAN_MASK
        )
        try:
            mask = np.load(location)
            return SlepianArbitrary(L, mask)
        except FileNotFoundError:
            logger.error(f"can not find the file: {config.SLEPIAN_MASK}")
            raise

    else:
        raise RuntimeError("no angle or file specified for Slepian region")
