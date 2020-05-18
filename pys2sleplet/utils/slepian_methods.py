from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.specific_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.specific_region.slepian_polar_cap import (
    SlepianPolarCap,
)
from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited

_file_location = Path(__file__).resolve()


def choose_slepian_method(L: int) -> SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    phi_min = np.deg2rad(config.PHI_MIN)
    phi_max = np.deg2rad(config.PHI_MAX)
    theta_min = np.deg2rad(config.THETA_MIN)
    theta_max = np.deg2rad(config.THETA_MAX)

    if is_polar_cap(phi_min, phi_max, theta_min, theta_max):
        logger.info("polar cap region detected")
        slepian = SlepianPolarCap(L, theta_max, config.ORDER)

    elif is_limited_lat_lon(phi_min, phi_max, theta_min, theta_max):
        logger.info("limited latitude longitude region detected")
        slepian = SlepianLimitLatLong(L, theta_min, theta_max, phi_min, phi_max)

    elif config.SLEPIAN_MASK:
        logger.info("no angles specified, looking for a file with mask")
        location = (
            _file_location.parents[2]
            / "data"
            / "slepian"
            / "arbitrary"
            / "masks"
            / config.SLEPIAN_MASK
        )
        try:
            mask = np.load(location)
            slepian = SlepianArbitrary(L, mask)
        except FileNotFoundError:
            logger.error(f"can not find the file: {config.SLEPIAN_MASK}")
            raise

    else:
        raise RuntimeError("no angle or file specified for Slepian region")

    return slepian


def apply_slepian_mask(function: Functions, slepian: SlepianFunctions) -> None:
    """
    when manipulating Slepian functions we need a map which has mask similar
    to that of the function so we can see the effect of convolutions etc
    """
    thetas, phis = ssht.sample_positions(config.L, Grid=True, Method="MWSS")
    whole_sphere_field = function.field

    if isinstance(slepian, SlepianPolarCap):
        condition = thetas <= slepian.theta_max
        region_field = np.where(condition, whole_sphere_field, 0)

    elif isinstance(slepian, SlepianLimitLatLong):
        condition = (
            (thetas >= slepian.theta_min)
            & (thetas <= slepian.theta_max)
            & (phis >= slepian.phi_min)
            & (phis <= slepian.phi_max)
        )
        region_field = np.where(condition, whole_sphere_field, 0)

    elif isinstance(slepian, SlepianArbitrary):
        region_field = whole_sphere_field * slepian.mask

    else:
        raise RuntimeError(f"{slepian} is not a valid slepian function type")

    flm = ensure_f_bandlimited(region_field, config.L, function.reality)
    function.multipole = flm
