from pathlib import Path
from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.harmonic_methods import boost_coefficient_resolution
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_sphere,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_slepian_data = _file_location.parents[1] / "data" / "slepian"


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
    eigenvectors: np.ndarray, L: int, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    helper function which integrates all of the slepian functionss
    """
    weight = calc_integration_weight(L)
    N = len(eigenvectors)
    output = np.zeros((N, N), dtype=np.complex128)
    for i, flm in enumerate(eigenvectors):
        for j, glm in enumerate(eigenvectors):
            # Hermitian matrix so can use symmetry
            if i <= j:
                f = ssht.inverse(flm, L, Method=SAMPLING_SCHEME)
                g = ssht.inverse(glm, L, Method=SAMPLING_SCHEME)
                output[j][i] = integrate_sphere(
                    L, f, g, weight, g_conj=True, mask=mask
                ).conj()
    fill_upper_triangle_of_hermitian_matrix(output)
    return output


def slepian_inverse(L: int, f_p: np.ndarray, slepian: SlepianFunctions) -> np.ndarray:
    """
    computes the Slepian inverse transform up to the Shannon number
    """
    n_theta, n_phi = ssht.sample_shape(L, Method=SAMPLING_SCHEME)
    f = np.zeros((n_theta, n_phi), dtype=np.complex128)
    for p in range(slepian.N):
        s_p = ssht.inverse(slepian.eigenvectors[p], L, Method=SAMPLING_SCHEME)
        f += f_p[p] * s_p
    return f


def slepian_forward(
    L: int,
    slepian: SlepianFunctions,
    f: Optional[np.ndarray] = None,
    flm: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    computes the Slepian forward transform for all coefficients
    """
    sd = SlepianDecomposition(L, slepian, f=f, flm=flm, mask=mask)
    return sd.decompose_all()


def _compute_s_p_omega(L: int, slepian: SlepianFunctions) -> np.ndarray:
    """
    method to calculate Sp(omega) for a given region
    """
    n_theta, n_phi = ssht.sample_shape(L, Method=SAMPLING_SCHEME)
    sp = np.zeros((slepian.N, n_theta, n_phi), dtype=np.complex128)
    for p in range(slepian.N):
        if p % L == 0:
            logger.info(f"compute Sp(omega) p={p+1}/{slepian.N}")
        sp[p] = ssht.inverse(slepian.eigenvectors[p], L, Method=SAMPLING_SCHEME)
    return sp


def compute_s_p_omega_prime(
    L: int, alpha: float, beta: float, slepian: SlepianFunctions
) -> complex:
    """
    method to pick out the desired angle from Sp(omega)
    """
    sp_omega = _compute_s_p_omega(L, slepian)
    p = ssht.theta_to_index(beta, L, Method=SAMPLING_SCHEME)
    q = ssht.phi_to_index(alpha, L, Method=SAMPLING_SCHEME)
    sp_omega_prime = sp_omega[:, p, q]
    # pad with zeros so it has the expected shape
    boost = L ** 2 - slepian.N
    return boost_coefficient_resolution(sp_omega_prime, boost)
