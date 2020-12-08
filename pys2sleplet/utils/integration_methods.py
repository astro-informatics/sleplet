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


def integrate_whole_sphere(f: np.ndarray, g: np.ndarray, weight: np.ndarray) -> complex:
    """
    computes the integration for the whole sphere
    """
    return (f * g * weight).sum()


def integrate_region_sphere(
    f: np.ndarray, g: np.ndarray, weight: np.ndarray, mask: np.ndarray
) -> complex:
    """
    computes the integration for a region of the sphere
    """
    return (f * g * weight * mask).sum()
