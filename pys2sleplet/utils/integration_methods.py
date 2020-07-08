from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def calc_integration_resolution(L: int) -> int:
    """
    calculate appropriate sample number for given L
    numbers based on test getting an atol=1e-5
    test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_matrix
    TODO: increase support to L=32 and beyond
    """
    sample_dict = {1: 287, 2: 210, 3: 159, 4: 157, 5: 129}

    for log_bandlimit, samples in sample_dict.items():
        if L < 2 ** log_bandlimit:
            return samples

    # otherwise use one sample
    return 1


def calc_integration_weight(L: int) -> np.ndarray:
    """
    computes the spherical Jacobian for the integration
    """
    theta_grid, phi_grid = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    delta_theta = np.ediff1d(theta_grid[:, 0]).mean()
    delta_phi = np.ediff1d(phi_grid[0]).mean()
    weight = np.sin(theta_grid) * delta_theta * delta_phi
    return weight


def integrate_sphere(
    L: int,
    resolution: int,
    flm: np.ndarray,
    glm: np.ndarray,
    weight: np.ndarray,
    flm_reality: bool = False,
    glm_reality: bool = False,
    flm_conj: bool = False,
    glm_conj: bool = False,
    mask_boosted: Optional[np.ndarray] = None,
) -> complex:
    """
    * method which computes the integration on the sphere for
      either the whole sphere or a region depended on the region variable
    * the function accepts arguments that control the reality of the inputs
      as well as the ability to have conjugates within the integral
    * the multipole resolutions are boosted prior to integration
    """
    if mask_boosted is not None:
        if mask_boosted.shape[0] - 1 == L:
            raise AttributeError(
                f"mismatch in mask shape {mask_boosted.shape} & resolution {resolution}"
            )

    f = invert_flm_boosted(flm, L, resolution, reality=flm_reality)
    g = invert_flm_boosted(glm, L, resolution, reality=glm_reality)

    if flm_conj:
        f = f.conj()
    if glm_conj:
        g = g.conj()

    integrand = f * g * weight

    if mask_boosted is not None:
        integrand = np.where(mask_boosted, integrand, 0)

    integration = integrand.sum()
    return integration
