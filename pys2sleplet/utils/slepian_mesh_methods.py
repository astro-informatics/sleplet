from typing import Optional

import numpy as np

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.meshes.classes.slepian_mesh_decomposition import (
    SlepianMeshDecomposition,
)
from pys2sleplet.utils.mesh_methods import mesh_forward, mesh_inverse
from pys2sleplet.utils.noise import create_mesh_noise


def clean_evals_and_evecs(
    eigendecomposition: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    need eigenvalues and eigenvectors to be in a certain format
    """
    # access values
    eigenvalues, eigenvectors = eigendecomposition

    # sort eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T
    return eigenvalues, eigenvectors


def compute_shannon(mesh: Mesh) -> int:
    """
    computes the effective Shannon number for a region of a mesh
    """
    num_basis_fun = mesh.basis_functions.shape[0]
    region_vertices = mesh.region.sum()
    total_vertices = mesh.region.shape[0]
    return round(region_vertices / total_vertices * num_basis_fun)


def create_slepian_mesh_noise(
    mesh: Mesh,
    slepian_eigenvalues: np.ndarray,
    slepian_functions: np.ndarray,
    shannon: int,
    slepian_signal: np.ndarray,
    snr_in: int,
) -> np.ndarray:
    """
    computes Gaussian white noise in Slepian space
    """
    u_i = mesh_forward(
        mesh.basis_functions,
        slepian_mesh_inverse(slepian_signal, mesh, slepian_functions, shannon),
    )
    n_i = create_mesh_noise(u_i, snr_in)
    return slepian_mesh_forward(
        mesh, slepian_eigenvalues, slepian_functions, shannon, u_i=n_i
    )


def slepian_mesh_forward(
    mesh: Mesh,
    slepian_eigenvalues: np.ndarray,
    slepian_functions: np.ndarray,
    shannon: int,
    u: Optional[np.ndarray] = None,
    u_i: Optional[np.ndarray] = None,
    mask: bool = False,
) -> np.ndarray:
    """
    computes the Slepian forward transform for all coefficients
    """
    sd = SlepianMeshDecomposition(
        mesh, slepian_eigenvalues, slepian_functions, shannon, u=u, u_i=u_i, mask=mask
    )
    return sd.decompose_all()


def slepian_mesh_inverse(
    f_p: np.ndarray,
    mesh: Mesh,
    slepian_functions: np.ndarray,
    shannon: int,
) -> np.ndarray:
    """
    computes the Slepian inverse transform on the mesh up to the Shannon number
    """
    p_idx = 0
    f_p_reshape = f_p[:shannon, np.newaxis]
    s_p = _compute_mesh_s_p_pixel(mesh.basis_functions, slepian_functions, shannon)
    return (f_p_reshape * s_p).sum(axis=p_idx)


def _compute_mesh_s_p_pixel(
    basis_functions: np.ndarray, slepian_functions: np.ndarray, shannon: int
) -> np.ndarray:
    """
    method to calculate Sp(omega) for a given region
    """
    sp = np.zeros((shannon, basis_functions.shape[1]))
    for p in range(shannon):
        sp[p] = mesh_inverse(basis_functions, slepian_functions[p])
    return sp
