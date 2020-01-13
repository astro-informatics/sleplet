import numpy as np

from .vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    SLEPIAN,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


def is_polar_cap(
    phi_min: float, phi_max: float, theta_min: float, theta_max: float
) -> bool:
    """
    circular mask at the north pole
    """
    if (
        phi_min == np.deg2rad(PHI_MIN_DEFAULT)
        and phi_max == np.deg2rad(PHI_MAX_DEFAULT)
        and theta_min == np.deg2rad(THETA_MIN_DEFAULT)
        and theta_max != np.deg2rad(THETA_MAX_DEFAULT)
    ):
        return True
    else:
        return False


def is_limited_lat_lon(
    phi_min: float, phi_max: float, theta_min: float, theta_max: float
) -> bool:
    """
    a region defined by angles, just need one to not be the default
    """
    if (
        not is_polar_cap(phi_min, phi_max, theta_min, theta_max)
        and phi_min != np.deg2rad(PHI_MIN_DEFAULT)
        or phi_max != np.deg2rad(PHI_MAX_DEFAULT)
        or theta_min != np.deg2rad(THETA_MIN_DEFAULT)
        or theta_max != np.deg2rad(THETA_MAX_DEFAULT)
    ):
        return True
    else:
        return False


def is_small_polar_cap(theta_max: float) -> bool:
    """
    assuming it is a polar cap
    small defined for visualisation purposes
    """
    if theta_max <= np.pi / 4:
        return True
    else:
        return False


def is_polar_gap() -> bool:
    """
    assuming it is a polar gap
    """
    if SLEPIAN["POLAR_GAP"]:
        return True
    else:
        return False
