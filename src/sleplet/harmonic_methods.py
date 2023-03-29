"""
methods to perform operations in Fourier space of the sphere or mesh
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from numpy.random import Generator

import sleplet
import sleplet._data.create_earth_flm
import sleplet.meshes.mesh

AFRICA_ALPHA = np.deg2rad(44)
AFRICA_BETA = np.deg2rad(87)
AFRICA_GAMMA = np.deg2rad(341)
SOUTH_AMERICA_ALPHA = np.deg2rad(54)
SOUTH_AMERICA_BETA = np.deg2rad(108)
SOUTH_AMERICA_GAMMA = np.deg2rad(63)


def _create_spherical_harmonic(L: int, ind: int) -> npt.NDArray[np.complex_]:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    flm = np.zeros(L**2, dtype=np.complex_)
    flm[ind] = 1
    return flm


def _boost_coefficient_resolution(
    flm: npt.NDArray[Any], boost: int
) -> npt.NDArray[Any]:
    """
    calculates a boost in resolution for given flm
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: npt.NDArray[np.complex_],
    L: int,
    resolution: int,
    *,
    reality: bool = False,
    spin: int = 0,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    performs the inverse harmonic transform
    """
    boost = resolution**2 - L**2
    flm = _boost_coefficient_resolution(flm, boost)
    return ssht.inverse(
        flm,
        resolution,
        Reality=reality,
        Spin=spin,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )


def _ensure_f_bandlimited(
    grid_fun: Callable[
        [npt.NDArray[np.float_], npt.NDArray[np.float_]], npt.NDArray[np.float_]
    ],
    L: int,
    *,
    reality: bool,
    spin: int,
) -> npt.NDArray[np.complex_]:
    """
    if the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it
    """
    thetas, phis = ssht.sample_positions(
        L, Grid=True, Method=sleplet._vars.SAMPLING_SCHEME
    )
    f = grid_fun(thetas, phis)
    return ssht.forward(
        f, L, Reality=reality, Spin=spin, Method=sleplet._vars.SAMPLING_SCHEME
    )


def _create_emm_vector(L: int) -> npt.NDArray[np.float_]:
    """
    create vector of m values for a given L
    """
    emm = np.zeros(2 * L * 2 * L)
    k = 0

    for ell in range(2 * L):
        M = 2 * ell + 1
        emm[k : k + M] = np.arange(-ell, ell + 1)
        k += M
    return emm


def compute_random_signal(
    L: int, rng: Generator, *, var_signal: float
) -> npt.NDArray[np.complex_]:
    """
    generates a normally distributed random signal of a
    complex signal with mean 0 and variance 1
    """
    return np.sqrt(var_signal / 2) * (
        rng.standard_normal(L**2) + 1j * rng.standard_normal(L**2)
    )


def mesh_forward(
    mesh: sleplet.meshes.mesh.Mesh, u: npt.NDArray[np.complex_ | np.float_]
) -> npt.NDArray[np.float_]:
    """
    computes the mesh forward transform from real space to harmonic space
    """
    u_i = np.zeros(mesh.mesh_eigenvalues.shape[0])
    for i, phi_i in enumerate(mesh.basis_functions):
        u_i[i] = sleplet._integration_methods.integrate_whole_mesh(
            mesh.vertices, mesh.faces, u, phi_i
        )
    return u_i


def mesh_inverse(
    mesh: sleplet.meshes.mesh.Mesh, u_i: npt.NDArray[np.complex_ | np.float_]
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the mesh inverse transform from harmonic space to real space
    """
    return (u_i[:, np.newaxis] * mesh.basis_functions).sum(axis=0)


def rotate_earth_to_south_america(
    earth_flm: npt.NDArray[np.complex_ | np.float_], L: int
) -> npt.NDArray[np.complex_]:
    """
    rotates the flms of the Earth to a view centered on South America
    """
    return ssht.rotate_flms(
        earth_flm, SOUTH_AMERICA_ALPHA, SOUTH_AMERICA_BETA, SOUTH_AMERICA_GAMMA, L
    )


def rotate_earth_to_africa(
    earth_flm: npt.NDArray[np.complex_ | np.float_], L: int
) -> npt.NDArray[np.complex_]:
    """
    rotates the flms of the Earth to a view centered on Africa
    """
    return ssht.rotate_flms(earth_flm, AFRICA_ALPHA, AFRICA_BETA, AFRICA_GAMMA, L)
