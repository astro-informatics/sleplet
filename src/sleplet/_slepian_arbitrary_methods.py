import typing

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import sleplet.meshes.mesh

_MACHINE_EPSILON = 1e-14


def clean_evals_and_evecs(
    eigendecomposition: tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
    """Need eigenvalues and eigenvectors to be in a certain format."""
    # access values
    eigenvalues_complex, eigenvectors = eigendecomposition

    # eigenvalues should be real
    eigenvalues = eigenvalues_complex.real

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].conj().T

    # ensure first element of each eigenvector is positive
    eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

    # find repeating eigenvalues and ensure orthorgonality
    pairs = np.where(np.abs(np.diff(eigenvalues)) < _MACHINE_EPSILON)[0] + 1
    eigenvectors[pairs] *= 1j

    return eigenvalues, eigenvectors


def compute_mesh_shannon(mesh: "sleplet.meshes.mesh.Mesh") -> int:
    """Compute the effective Shannon number for a region of a mesh."""
    num_basis_fun = mesh.mesh_eigenvalues.shape[0]
    region_vertices = mesh.mesh_region.sum()
    total_vertices = mesh.mesh_region.shape[0]
    return round(region_vertices / total_vertices * num_basis_fun)
