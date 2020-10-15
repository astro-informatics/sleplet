from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.harmonic_methods import invert_flm_boosted


def calc_integration_resolution(L: int) -> int:
    """
    calculate appropriate sample number for given L
    """
    return L * 2


def calc_integration_weight(L: int) -> np.ndarray:
    """
    computes the spherical Jacobian for the integration
    """
    thetas, phis = ssht.sample_positions(L, Grid=True)
    delta_theta = np.ediff1d(thetas[:, 0]).mean()
    delta_phi = np.ediff1d(phis[0]).mean()
    return np.sin(thetas) * delta_theta * delta_phi


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
    flm_spin: int = 0,
    glm_spin: int = 0,
    mask_boosted: Optional[np.ndarray] = None,
) -> complex:
    """
    * method which computes the integration on the sphere for
      either the whole sphere or a region depended on the region variable
    * the function accepts arguments that control the reality of the inputs
      as well as the ability to have conjugates within the integral
    * the coefficients resolutions are boosted prior to integration
    """
    if mask_boosted is not None and mask_boosted.shape != ssht.sample_shape(resolution):
        raise AttributeError(
            f"mismatch in mask shape {mask_boosted.shape} & resolution {resolution}"
        )

    f = invert_flm_boosted(flm, L, resolution, reality=flm_reality, spin=flm_spin)
    g = invert_flm_boosted(glm, L, resolution, reality=glm_reality, spin=glm_spin)

    if flm_conj:
        f = f.conj()
    if glm_conj:
        g = g.conj()

    integrand = f * g * weight

    if mask_boosted is not None:
        integrand = np.where(mask_boosted, integrand, 0)

    return integrand.sum()
