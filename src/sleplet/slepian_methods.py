"""
methods to work with Slepian coefficients
"""
import numpy as np
import pyssht as ssht
from numpy import typing as npt

import sleplet
import sleplet._vars
import sleplet.harmonic_methods
import sleplet.meshes._mesh_slepian_decomposition
import sleplet.meshes.mesh_slepian
import sleplet.region
import sleplet.slepian._slepian_decomposition
import sleplet.slepian.slepian_arbitrary
import sleplet.slepian.slepian_functions
import sleplet.slepian.slepian_limit_lat_lon
import sleplet.slepian.slepian_polar_cap


def choose_slepian_method(
    L: int, region: sleplet.region.Region
) -> sleplet.slepian.slepian_functions.SlepianFunctions:
    """
    initialise Slepian object depending on input
    """
    match region.region_type:
        case "polar":
            sleplet.logger.info("polar cap region detected")
            return sleplet.slepian.slepian_polar_cap.SlepianPolarCap(
                L, region.theta_max, gap=region.gap
            )

        case "lim_lat_lon":
            sleplet.logger.info("limited latitude longitude region detected")
            return sleplet.slepian.slepian_limit_lat_lon.SlepianLimitLatLon(
                L,
                theta_min=region.theta_min,
                theta_max=region.theta_max,
                phi_min=region.phi_min,
                phi_max=region.phi_max,
            )

        case "arbitrary":
            sleplet.logger.info("mask specified in file detected")
            return sleplet.slepian.slepian_arbitrary.SlepianArbitrary(
                L, region.mask_name
            )

        case _:
            raise ValueError(f"{region.region_type} is an invalid region type")


def slepian_inverse(
    f_p: npt.NDArray[np.complex_ | np.float_],
    L: int,
    slepian: sleplet.slepian.slepian_functions.SlepianFunctions,
) -> npt.NDArray[np.complex_]:
    """
    computes the Slepian inverse transform up to the Shannon number
    """
    f_p_reshape = f_p[: slepian.N, np.newaxis, np.newaxis]
    s_p = _compute_s_p_omega(L, slepian)
    return (f_p_reshape * s_p).sum(axis=0)


def slepian_forward(
    L: int,
    slepian: sleplet.slepian.slepian_functions.SlepianFunctions,
    *,
    f: npt.NDArray[np.complex_] | None = None,
    flm: npt.NDArray[np.complex_ | np.float_] | None = None,
    mask: npt.NDArray[np.float_] | None = None,
    n_coeffs: int | None = None,
) -> npt.NDArray[np.complex_]:
    """
    computes the Slepian forward transform for all coefficients
    """
    sd = sleplet.slepian._slepian_decomposition.SlepianDecomposition(
        L, slepian, f=f, flm=flm, mask=mask
    )
    n_coeffs = slepian.N if n_coeffs is None else n_coeffs
    return sd.decompose_all(n_coeffs)


def _compute_s_p_omega(
    L: int, slepian: sleplet.slepian.slepian_functions.SlepianFunctions
) -> npt.NDArray[np.complex_]:
    """
    method to calculate Sp(omega) for a given region
    """
    n_theta, n_phi = ssht.sample_shape(L, Method=sleplet._vars.SAMPLING_SCHEME)
    sp = np.zeros((slepian.N, n_theta, n_phi), dtype=np.complex_)
    for p in range(slepian.N):
        if p % L == 0:
            sleplet.logger.info(f"compute Sp(omega) p={p+1}/{slepian.N}")
        sp[p] = ssht.inverse(
            slepian.eigenvectors[p], L, Method=sleplet._vars.SAMPLING_SCHEME
        )
    return sp


def _compute_s_p_omega_prime(
    L: int,
    alpha: float,
    beta: float,
    slepian: sleplet.slepian.slepian_functions.SlepianFunctions,
) -> npt.NDArray[np.complex_]:
    """
    method to pick out the desired angle from Sp(omega)
    """
    sp_omega = _compute_s_p_omega(L, slepian)
    p = ssht.theta_to_index(beta, L, Method=sleplet._vars.SAMPLING_SCHEME)
    q = ssht.phi_to_index(alpha, L, Method=sleplet._vars.SAMPLING_SCHEME)
    sp_omega_prime = sp_omega[:, p, q]
    # pad with zeros so it has the expected shape
    boost = L**2 - slepian.N
    return sleplet.harmonic_methods._boost_coefficient_resolution(sp_omega_prime, boost)


def slepian_mesh_forward(
    mesh_slepian: sleplet.meshes.mesh_slepian.MeshSlepian,
    *,
    u: npt.NDArray[np.complex_ | np.float_] | None = None,
    u_i: npt.NDArray[np.complex_ | np.float_] | None = None,
    mask: bool = False,
    n_coeffs: int | None = None,
) -> npt.NDArray[np.float_]:
    """
    computes the Slepian forward transform for all coefficients
    """
    sd = sleplet.meshes._mesh_slepian_decomposition.MeshSlepianDecomposition(
        mesh_slepian,
        u=u,
        u_i=u_i,
        mask=mask,
    )
    n_coeffs = mesh_slepian.N if n_coeffs is None else n_coeffs
    return sd.decompose_all(n_coeffs)


def slepian_mesh_inverse(
    mesh_slepian: sleplet.meshes.mesh_slepian.MeshSlepian,
    f_p: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the Slepian inverse transform on the mesh up to the Shannon number
    """
    f_p_reshape = f_p[: mesh_slepian.N, np.newaxis]
    s_p = _compute_mesh_s_p_pixel(mesh_slepian)
    return (f_p_reshape * s_p).sum(axis=0)


def _compute_mesh_s_p_pixel(
    mesh_slepian: sleplet.meshes.mesh_slepian.MeshSlepian,
) -> npt.NDArray[np.float_]:
    """
    method to calculate Sp(omega) for a given region
    """
    sp = np.zeros((mesh_slepian.N, mesh_slepian.mesh.vertices.shape[0]))
    for p in range(mesh_slepian.N):
        sp[p] = sleplet.harmonic_methods.mesh_inverse(
            mesh_slepian.mesh, mesh_slepian.slepian_functions[p]
        )
    return sp
