from pathlib import Path

import numpy as np

from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix


def calculate_high_L_matrix(file_loc: Path, L: int, L_ranges: list[int]) -> np.ndarray:
    """
    splits up and calculates intermediate matrices for higher L
    """
    D = np.zeros((L**2, L**2), dtype=np.complex_)
    for i in range(len(L_ranges) - 1):
        L_min = L_ranges[i]
        L_max = L_ranges[i + 1]
        x = np.load(file_loc / f"D_min{L_min}_max{L_max}.npy")
        D += x

    # fill in remaining triangle section
    fill_upper_triangle_of_hermitian_matrix(D)
    return D


def clean_evals_and_evecs(
    eigendecomposition: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    need eigenvalues and eigenvectors to be in a certain format
    """
    # access values
    eigenvalues, eigenvectors = eigendecomposition

    # eigenvalues should be real
    eigenvalues = eigenvalues.real

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].conj().T

    # ensure first element of each eigenvector is positive
    eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

    # find repeating eigenvalues and ensure orthorgonality
    pairs = np.where(np.abs(np.diff(eigenvalues)) < 1e-14)[0] + 1
    eigenvectors[pairs] *= 1j

    return eigenvalues, eigenvectors


def compute_mesh_shannon(mesh: Mesh) -> int:
    """
    computes the effective Shannon number for a region of a mesh
    """
    num_basis_fun = mesh.mesh_eigenvalues.shape[0]
    region_vertices = mesh.region.sum()
    total_vertices = mesh.region.shape[0]
    return round(region_vertices / total_vertices * num_basis_fun)
