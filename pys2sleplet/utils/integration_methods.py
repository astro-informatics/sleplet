from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.harmonic_methods import invert_flm
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import create_mask_region
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def calc_integration_resolution(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    sample_dict = {
        1: 1801,
        2: 901,
        3: 451,
        4: 226,
        5: 113,
        6: 57,
        7: 29,
        8: 15,
        9: 8,
        10: 4,
        11: 2,
    }

    for log_bandlimit, samples in sample_dict.items():
        if L < 2 ** log_bandlimit:
            return samples

    # above L = 2048 just use 1 sample
    return 1


def _integration_weight(L: int) -> np.ndarray:
    """
    computes the spherical Jacobian for the integration
    """
    theta_grid, phi_grid = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    delta_theta = np.ediff1d(theta_grid[:, 0]).mean()
    delta_phi = np.ediff1d(phi_grid[0]).mean()
    weight = np.sin(theta_grid) * delta_theta * delta_phi
    return weight


def _integration_helper(
    L: int,
    flm: np.ndarray,
    glm: np.ndarray,
    flm_reality: bool,
    glm_reality: bool,
    flm_conj: bool,
    glm_conj: bool,
    region: Optional[Region] = None,
) -> float:
    """
    * helper method which computes the integration on the sphere for
      either the whole sphere or a region depended on the region variable
    * the function accepts arguments that control the reality of the inputs
      as well as the ability to have conjugates within the integral
    * the multipole resolutions are boosted prior to integration
    """
    resolution = calc_integration_resolution(L)

    if region is not None:
        mask = create_mask_region(resolution, region)
    else:
        mask = None

    weight = _integration_weight(resolution)
    f = invert_flm(flm, L, reality=flm_reality, resolution=resolution)
    g = invert_flm(glm, L, reality=glm_reality, resolution=resolution)

    if flm_conj:
        f = f.conj()
    if glm_conj:
        g = g.conj()

    integrand = f * g * weight

    if mask is not None:
        integrand = np.where(mask, integrand, 0)

    integration = integrand.sum()
    return integration


def integrate_whole_sphere(
    L: int,
    flm: np.ndarray,
    glm: np.ndarray,
    flm_reality: bool = False,
    glm_reality: bool = False,
    flm_conj: bool = False,
    glm_conj: bool = False,
) -> float:
    """
    integrates over the whole sphere using the helper method
    """
    integration = _integration_helper(
        L, flm, glm, flm_reality, glm_reality, flm_conj, glm_conj
    )
    return integration


def integrate_region_sphere(
    L: int,
    flm: np.ndarray,
    glm: np.ndarray,
    region: Region,
    flm_reality: bool = False,
    glm_reality: bool = False,
    flm_conj: bool = False,
    glm_conj: bool = False,
) -> float:
    """
    * integrates over a region of the sphere using the helper method
    * the region must be calculated for the integration resolution
    """
    integration = _integration_helper(
        L, flm, glm, flm_reality, glm_reality, flm_conj, glm_conj, region=region
    )
    return integration
