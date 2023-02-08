from functools import reduce

import numpy as np
import pyssht as ssht
from numpy import typing as npt

from sleplet.utils.vars import SAMPLING_SCHEME


def calc_integration_weight(L: int) -> npt.NDArray:
    """
    computes the spherical Jacobian for the integration
    """
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    delta_theta = np.ediff1d(thetas[:, 0]).mean()
    delta_phi = np.ediff1d(phis[0]).mean()
    return np.sin(thetas) * delta_theta * delta_phi


def integrate_whole_sphere(weight: npt.NDArray, *functions: npt.NDArray) -> complex:
    """
    computes the integration for the whole sphere
    """
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight).sum()


def integrate_region_sphere(
    mask: npt.NDArray, weight: npt.NDArray, *functions: npt.NDArray
) -> complex:
    """
    computes the integration for a region of the sphere
    """
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight * mask).sum()


def integrate_whole_mesh(
    vertices: npt.NDArray, faces: npt.NDArray, *functions: npt.NDArray
) -> float:
    """
    computes the integral of functions on the vertices
    """
    multiplied_inputs = _multiply_args(*functions)
    return multiplied_inputs.sum()


def integrate_region_mesh(
    mask: npt.NDArray,
    vertices: npt.NDArray,
    faces: npt.NDArray,
    *functions: npt.NDArray,
) -> float:
    """
    computes the integral of a region of functions on the vertices
    """
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * mask).sum()


def _multiply_args(*args: npt.NDArray) -> npt.NDArray:
    """
    method to multiply an unknown number of arguments
    """
    return reduce((lambda x, y: x * y), args)
