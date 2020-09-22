from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_sphere,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region


def choose_slepian_method(L: int, region: Region) -> SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    if region.region_type == "polar":
        logger.info("polar cap region detected")
        slepian = SlepianPolarCap(L, region.theta_max, gap=region.gap)

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


def integrate_whole_matrix_slepian_functions(
    eigenvectors: np.ndarray, L: int, resolution: int, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    helper function which integrates all of the slepian functionss
    """
    weight = calc_integration_weight(resolution)
    N = len(eigenvectors)
    output = np.zeros((N, N), dtype=np.complex128)
    for i, flm in enumerate(eigenvectors):
        for j, glm in enumerate(eigenvectors):
            # Hermitian matrix so can use symmetry
            if i <= j:
                output[j][i] = integrate_sphere(
                    L, resolution, flm, glm, weight, glm_conj=True, mask_boosted=mask
                ).conj()
    fill_upper_triangle_of_hermitian_matrix(output)
    return output


def slepian_inverse(L: int, f_p: np.ndarray, slepian: SlepianFunctions) -> np.ndarray:
    """
    computes the Slepian inverse transform up to the Shannon number
    """
    n_theta, n_phi = ssht.sample_shape(L)
    f = np.zeros((n_theta, n_phi), dtype=np.complex128)
    for p in range(slepian.N):
        s_p = ssht.inverse(slepian.eigenvectors[p], L)
        f += f_p[p] * s_p
    return f


def slepian_forward(
    L: int, flm: np.ndarray, slepian: SlepianFunctions, method: str = "harmonic_sum"
) -> np.ndarray:
    """
    computes the Slepian forward transform for all coefficients
    """
    sd = SlepianDecomposition(L, flm, slepian)
    return sd.decompose_all(method=method)
