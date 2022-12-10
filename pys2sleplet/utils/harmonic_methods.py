from collections.abc import Callable

import numpy as np
import pyssht as ssht
from numpy.random import Generator

from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.integration_methods import integrate_whole_mesh
from sleplet.utils.vars import SAMPLING_SCHEME


def create_spherical_harmonic(L: int, ind: int) -> np.ndarray:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    flm = np.zeros(L**2, dtype=np.complex_)
    flm[ind] = 1
    return flm


def boost_coefficient_resolution(flm: np.ndarray, boost: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: np.ndarray, L: int, resolution: int, *, reality: bool = False, spin: int = 0
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    boost = resolution**2 - L**2
    flm = boost_coefficient_resolution(flm, boost)
    return ssht.inverse(
        flm, resolution, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME
    )


def ensure_f_bandlimited(
    grid_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    L: int,
    reality: bool,
    spin: int,
) -> np.ndarray:
    """
    if the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it
    """
    thetas, phis = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    f = grid_fun(thetas, phis)
    return ssht.forward(f, L, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME)


def create_emm_vector(L: int) -> np.ndarray:
    """
    create vector of m values for a given L
    """
    emm = np.zeros(2 * L * 2 * L)
    k = 0

    for l in range(2 * L):
        M = 2 * l + 1
        emm[k : k + M] = np.arange(-l, l + 1)
        k += M
    return emm


def compute_random_signal(L: int, rng: Generator, *, var_signal: float) -> np.ndarray:
    """
    generates a normally distributed random signal of a
    complex signal with mean 0 and variance 1
    """
    return np.sqrt(var_signal / 2) * (
        rng.standard_normal(L**2) + 1j * rng.standard_normal(L**2)
    )


def mesh_forward(mesh: Mesh, u: np.ndarray) -> np.ndarray:
    """
    computes the mesh forward transform from real space to harmonic space
    """
    u_i = np.zeros(mesh.mesh_eigenvalues.shape[0])
    for i, phi_i in enumerate(mesh.basis_functions):
        u_i[i] = integrate_whole_mesh(mesh.vertices, mesh.faces, u, phi_i)
    return u_i


def mesh_inverse(mesh: Mesh, u_i: np.ndarray) -> np.ndarray:
    """
    computes the mesh inverse transform from harmonic space to real space
    """
    return (u_i[:, np.newaxis] * mesh.basis_functions).sum(axis=0)
