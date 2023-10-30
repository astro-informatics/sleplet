import functools
import typing

import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet._vars


def calc_integration_weight(L: int) -> npt.NDArray[np.float_]:
    """Compute the spherical Jacobian for the integration."""
    thetas, phis = ssht.sample_positions(
        L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    delta_theta = np.ediff1d(thetas[:, 0]).mean()
    delta_phi = np.ediff1d(phis[0]).mean()
    return np.sin(thetas) * delta_theta * delta_phi


def integrate_whole_sphere(
    weight: npt.NDArray[np.float_],
    *functions: npt.NDArray[np.complex_],
) -> complex:
    """Compute the integration for the whole sphere."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight).sum()


def integrate_region_sphere(
    mask: npt.NDArray[np.float_],
    weight: npt.NDArray[np.float_],
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> complex:
    """Compute the integration for a region of the sphere."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight * mask).sum()


def integrate_whole_mesh(
    vertices: npt.NDArray[np.float_],  # noqa: ARG001
    faces: npt.NDArray[np.int_],  # noqa: ARG001
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> float:
    """Compute the integral of functions on the vertices."""
    multiplied_inputs = _multiply_args(*functions)
    return multiplied_inputs.sum()


def integrate_region_mesh(
    mask: npt.NDArray[np.bool_],
    vertices: npt.NDArray[np.float_],  # noqa: ARG001
    faces: npt.NDArray[np.int_],  # noqa: ARG001
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> float:
    """Compute the integral of a region of functions on the vertices."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * mask).sum()


def _multiply_args(*args: npt.NDArray[typing.Any]) -> npt.NDArray[typing.Any]:
    """Multiply an unknown number of arguments."""
    return functools.reduce((lambda x, y: x * y), args)
