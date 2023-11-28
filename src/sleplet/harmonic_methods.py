"""Methods to perform operations in Fourier space of the sphere or mesh."""
import collections
import typing

import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet._integration_methods
import sleplet._vars
import sleplet.meshes.mesh

_AFRICA_ALPHA = np.deg2rad(44)
_AFRICA_BETA = np.deg2rad(87)
_AFRICA_GAMMA = np.deg2rad(341)
_SOUTH_AMERICA_ALPHA = np.deg2rad(54)
_SOUTH_AMERICA_BETA = np.deg2rad(108)
_SOUTH_AMERICA_GAMMA = np.deg2rad(63)


def _create_spherical_harmonic(L: int, ind: int) -> npt.NDArray[np.complex_]:
    """Create a spherical harmonic in harmonic space for the given index."""
    flm = np.zeros(L**2, dtype=np.complex_)
    flm[ind] = 1
    return flm


def _boost_coefficient_resolution(
    flm: npt.NDArray[typing.Any],
    boost: int,
) -> npt.NDArray[typing.Any]:
    """Calculate a boost in resolution for given flm."""
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
    Upsamples the signal and performs the inverse harmonic transform .

    Args:
        flm: The spherical harmonic coefficients.
        L: The spherical harmonic bandlimit.
        resolution: The output resolution of the field values.
        reality: Whether the given spherical signal is real or not.
        spin: The value of the spin.

    Returns:
        The boosted field value.
    """
    boost = resolution**2 - L**2
    flm = _boost_coefficient_resolution(flm, boost)
    return ssht.inverse(
        flm,
        resolution,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=reality,
        Spin=spin,
    )


def _ensure_f_bandlimited(
    grid_fun: collections.abc.Callable[
        [npt.NDArray[np.float_], npt.NDArray[np.float_]],
        npt.NDArray[np.float_],
    ],
    L: int,
    *,
    reality: bool,
    spin: int,
) -> npt.NDArray[np.complex_]:
    """
    If the function created is created in pixel space rather than harmonic
    space then need to transform it into harmonic space first before using it.
    """
    thetas, phis = ssht.sample_positions(
        L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    f = grid_fun(thetas, phis)
    return ssht.forward(
        f,
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=reality,
        Spin=spin,
    )


def _create_emm_vector(L: int) -> npt.NDArray[np.float_]:
    """Create vector of m values for a given L."""
    emm = np.zeros(2 * L * 2 * L)
    k = 0

    for ell in range(2 * L):
        M = 2 * ell + 1
        emm[k : k + M] = np.arange(-ell, ell + 1)
        k += M
    return emm


def compute_random_signal(
    L: int,
    rng: np.random.Generator,
    *,
    var_signal: float,
) -> npt.NDArray[np.complex_]:
    """
    Generate a normally distributed random signal of a
    complex signal with mean `0` and variance `1`.

    Args:
        L: The spherical harmonic bandlimit.
        rng: The random number generator object.
        var_signal: The variance of the signal.

    Returns:
        The coefficients of a random signal on the sphere.
    """
    return np.sqrt(var_signal / 2) * (
        rng.standard_normal(L**2) + 1j * rng.standard_normal(L**2)
    )


def mesh_forward(
    mesh: "sleplet.meshes.mesh.Mesh",
    u: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.float_]:
    """
    Compute the mesh forward transform from pixel space to Fourier space.

    Args:
        mesh: The given mesh object.
        u: The signal field value on the mesh.

    Returns:
        The basis functions of the mesh in Fourier space.
    """
    u_i = np.zeros(mesh.mesh_eigenvalues.shape[0])
    for i, phi_i in enumerate(mesh.basis_functions):
        u_i[i] = sleplet._integration_methods.integrate_whole_mesh(
            mesh.vertices,
            mesh.faces,
            u,
            phi_i,
        )
    return u_i


def mesh_inverse(
    mesh: "sleplet.meshes.mesh.Mesh",
    u_i: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    Compute the mesh inverse transform from Fourier space to pixel space.

    Args:
        mesh: The given mesh object.
        u_i: The Fourier coefficients on the mesh.

    Returns:
        The values on the mesh in pixel space.
    """
    return (u_i[:, np.newaxis] * mesh.basis_functions).sum(axis=0)


def rotate_earth_to_south_america(
    earth_flm: npt.NDArray[np.complex_ | np.float_],
    L: int,
) -> npt.NDArray[np.complex_]:
    """
    Rotates the harmonic coefficients of the Earth to a view centered on South America.

    Args:
        earth_flm: The spherical harmonic coefficients of the Earth.
        L: The spherical harmonic bandlimit.

    Returns:
        The spherical harmonic coefficients of the Earth centered on South America.
    """
    return ssht.rotate_flms(
        earth_flm,
        _SOUTH_AMERICA_ALPHA,
        _SOUTH_AMERICA_BETA,
        _SOUTH_AMERICA_GAMMA,
        L,
    )


def rotate_earth_to_africa(
    earth_flm: npt.NDArray[np.complex_ | np.float_],
    L: int,
) -> npt.NDArray[np.complex_]:
    """
    Rotates the harmonic coefficients of the Earth to a view centered on Africa.

    Args:
        earth_flm: The spherical harmonic coefficients of the Earth.
        L: The spherical harmonic bandlimit.

    Returns:
        The spherical harmonic coefficients of the Earth centered on Africa.
    """
    return ssht.rotate_flms(earth_flm, _AFRICA_ALPHA, _AFRICA_BETA, _AFRICA_GAMMA, L)
