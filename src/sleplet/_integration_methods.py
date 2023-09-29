import functools
import typing

import numpy as np
import numpy.typing as npt

import s2fft

import sleplet._vars


def calc_integration_weight(L: int) -> npt.NDArray[np.float_]:
    """Computes the spherical Jacobian for the integration."""
    thetas = np.tile(
        s2fft.samples.thetas(L, sampling=sleplet._vars.SAMPLING_SCHEME),
        (s2fft.samples.nphi_equiang(L, sampling=sleplet._vars.SAMPLING_SCHEME), 1),
    ).T
    phis = np.tile(
        s2fft.samples.phis_equiang(L, sampling=sleplet._vars.SAMPLING_SCHEME),
        (s2fft.samples.ntheta(L, sampling=sleplet._vars.SAMPLING_SCHEME), 1),
    )
    delta_theta = np.ediff1d(thetas[:, 0]).mean()
    delta_phi = np.ediff1d(phis[0]).mean()
    return np.sin(thetas) * delta_theta * delta_phi


def integrate_whole_sphere(
    weight: npt.NDArray[np.float_],
    *functions: npt.NDArray[np.complex_],
) -> complex:
    """Computes the integration for the whole sphere."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight).sum()


def integrate_region_sphere(
    mask: npt.NDArray[np.float_],
    weight: npt.NDArray[np.float_],
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> complex:
    """Computes the integration for a region of the sphere."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * weight * mask).sum()


def integrate_whole_mesh(
    vertices: npt.NDArray[np.float_],  # noqa: ARG001
    faces: npt.NDArray[np.int_],  # noqa: ARG001
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> float:
    """Computes the integral of functions on the vertices."""
    multiplied_inputs = _multiply_args(*functions)
    return multiplied_inputs.sum()


def integrate_region_mesh(
    mask: npt.NDArray[np.bool_],
    vertices: npt.NDArray[np.float_],  # noqa: ARG001
    faces: npt.NDArray[np.int_],  # noqa: ARG001
    *functions: npt.NDArray[np.complex_ | np.float_],
) -> float:
    """Computes the integral of a region of functions on the vertices."""
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * mask).sum()


def _multiply_args(*args: npt.NDArray[typing.Any]) -> npt.NDArray[typing.Any]:
    """Method to multiply an unknown number of arguments."""
    return functools.reduce((lambda x, y: x * y), args)
