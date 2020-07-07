from typing import Optional

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.integration_methods import integrate_sphere
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region


def choose_slepian_method(L: int, region: Region) -> SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    if region.region_type == "polar":
        logger.info("polar cap region detected")
        slepian = SlepianPolarCap(
            L, region.theta_max, order=region.order, gap=region.gap
        )

    elif region.region_type == "lim_lat_lon":
        logger.info("limited latitude longitude region detected")
        slepian = SlepianLimitLatLon(
            L,
            theta_min=region.theta_min,
            theta_max=region.theta_max,
            phi_min=region.phi_min,
            phi_max=region.phi_max,
        )

    elif region.region_type == "arbitrary":
        logger.info("mask specified in file detected")
        slepian = SlepianArbitrary(L, region.mask_name)

    return slepian


def integrate_two_slepian_functions_per_rank(
    eigenvectors: np.ndarray,
    L: int,
    resolution: int,
    rank1: int,
    rank2: int,
    mask: Optional[np.ndarray] = None,
) -> complex:
    """
    helper function which integrates two slepian functions of given ranks
    """
    flm = eigenvectors[rank1]
    glm = eigenvectors[rank2]
    output = integrate_sphere(L, resolution, flm, glm, glm_conj=True, mask_boosted=mask)
    return output


def integrate_whole_matrix_slepian_functions(
    eigenvectors: np.ndarray, L: int, resolution: int, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    helper function which integrates all of the slepian functionss
    """
    N = len(eigenvectors)
    output = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(eigenvectors):
        for j, glm in enumerate(eigenvectors):
            # Hermitian matrix so can use symmetry
            if i <= j:
                output[j][i] = integrate_sphere(
                    L, resolution, flm, glm, glm_conj=True, mask_boosted=mask
                ).conj()
    fill_upper_triangle_of_hermitian_matrix(output)
    return output
