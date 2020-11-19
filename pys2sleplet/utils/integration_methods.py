from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.vars import SAMPLING_SCHEME


def calc_integration_weight(L: int) -> np.ndarray:
    """
    computes the spherical Jacobian for the integration
    """
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    delta_theta = np.ediff1d(thetas[:, 0]).mean()
    delta_phi = np.ediff1d(phis[0]).mean()
    return np.sin(thetas) * delta_theta * delta_phi


def integrate_sphere(
    L: int,
    f: np.ndarray,
    g: np.ndarray,
    weight: np.ndarray,
    f_conj: bool = False,
    g_conj: bool = False,
    mask: Optional[np.ndarray] = None,
) -> complex:
    """
    * method which computes the integration on the sphere for
      either the whole sphere or a region depended on the region variable
    * the function accepts arguments that control the reality of the inputs
      as well as the ability to have conjugates within the integral
    """
    if isinstance(mask, np.ndarray) and mask.shape != ssht.sample_shape(
        L, Method=SAMPLING_SCHEME
    ):
        raise AttributeError(f"mismatch in mask shape {mask.shape} & bandlimit {L}")

    if f_conj:
        f = f.conj()
    if g_conj:
        g = g.conj()

    integrand = f * g * weight

    if isinstance(mask, np.ndarray):
        integrand = np.where(mask, integrand, 0)

    return integrand.sum()
