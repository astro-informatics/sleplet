from typing import Optional

import numpy as np
import pyssht as ssht


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
    flm: np.ndarray,
    glm: np.ndarray,
    weight: np.ndarray,
    flm_reality: bool = False,
    glm_reality: bool = False,
    flm_conj: bool = False,
    glm_conj: bool = False,
    flm_spin: int = 0,
    glm_spin: int = 0,
    mask: Optional[np.ndarray] = None,
) -> complex:
    """
    * method which computes the integration on the sphere for
      either the whole sphere or a region depended on the region variable
    * the function accepts arguments that control the reality of the inputs
      as well as the ability to have conjugates within the integral
    """
    if mask is not None and mask.shape != ssht.sample_shape(L):
        raise AttributeError(f"mismatch in mask shape {mask.shape} & bandlimit {L}")

    f = ssht.inverse(flm, L, Reality=flm_reality, Spin=flm_spin)
    g = ssht.inverse(glm, L, Reality=glm_reality, Spin=glm_spin)

    if flm_conj:
        f = f.conj()
    if glm_conj:
        g = g.conj()

    integrand = f * g * weight

    if mask is not None:
        integrand = np.where(mask, integrand, 0)

    return integrand.sum()
