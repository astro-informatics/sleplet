import numpy as np

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import choose_slepian_method, slepian_forward


def get_shannon(L: int, theta_max: int) -> int:
    """
    computes the Shannon number
    """
    slepian = SlepianPolarCap(L, np.deg2rad(theta_max))
    return slepian.N


def earth_region_harmonic_coefficients(L: int, theta_max: int) -> np.ndarray:
    """
    harmonic coefficients of the Earth for the polar cap region
    """
    region = Region(theta_max=np.deg2rad(theta_max))
    earth = Earth(L, region=region)
    coefficients = np.abs(earth.multipole)
    coefficients[::-1].sort()
    return coefficients


def earth_region_slepian_coefficients(
    L: int, theta_max: int, method: str = "harmonic_sum"
) -> np.ndarray:
    """
    computes the Slepian coefficients
    """
    region = Region(theta_max=np.deg2rad(theta_max))
    earth = Earth(L, region=region)
    slepian = choose_slepian_method(L, region)
    return np.abs(slepian_forward(L, earth.multipole, slepian, method=method))
