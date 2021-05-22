import glob
from pathlib import Path

import numpy as np
from box import Box
from igl import adjacency_matrix, all_pairs_distances, massmatrix, read_triangle_mesh
from numpy import linalg as LA

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"
MESHES: set[str] = {
    Path(x.removesuffix(".toml")).stem
    for x in glob.glob(str(_meshes_path / "regions" / "*.toml"))
}


def _read_toml(mesh_name: str) -> Box:
    """
    reads in the given mesh region settings file
    """
    return Box.from_toml(filename=_meshes_path / "regions" / f"{mesh_name}.toml")


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    data = _read_toml(mesh_name)
    vertices, faces = read_triangle_mesh(str(_meshes_path / "polygons" / data.FILENAME))
    return vertices, faces


def create_mesh_region(mesh_name: str, vertices: np.ndarray) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    data = _read_toml(mesh_name)
    return (
        (vertices[:, 0] > data.XMIN)
        & (vertices[:, 0] < data.XMAX)
        & (vertices[:, 1] > data.YMIN)
        & (vertices[:, 1] < data.YMAX)
        & (vertices[:, 2] > data.ZMIN)
        & (vertices[:, 2] < data.ZMAX)
    )


def _weighting_function(
    D: np.ndarray, A: np.ndarray, vertices: np.ndarray, theta: int, knn: int
) -> np.ndarray:
    """
    thresholded Gaussian kernel weighting function
    """
    W = np.exp(
        -all_pairs_distances(vertices, vertices, squared=True) / (2 * theta ** 2)
    )
    knn_threshold = D <= knn
    return W * A * knn_threshold


def _graph_laplacian(
    vertices: np.ndarray, faces: np.ndarray, theta: int, knn: int
) -> np.ndarray:
    """
    computes the graph laplacian L = D - W
    where D is the degree matrix and W is the weighting function
    """
    rows = 0
    A = adjacency_matrix(faces)
    D = np.diagflat(A.sum(axis=rows))
    W = _weighting_function(D, A, vertices, theta, knn)
    return D - W


def mesh_eigendecomposition(
    name: str, vertices: np.ndarray, faces: np.ndarray, theta: int = 1, knn: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of the mesh represented as a graph
    if already computed then it loads the data
    """
    logger.info(f"finding {vertices.shape[0]} basis functions of mesh")
    eigd_loc = _meshes_path / "basis_functions" / f"{name}_t{theta}_k{knn}"
    eval_loc = eigd_loc / "eigenvalues.npy"
    evec_loc = eigd_loc / "eigenvectors.npy"
    if eval_loc.exists() and evec_loc.exists():
        logger.info("binaries found - loading...")
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
    else:
        laplacian = _graph_laplacian(vertices, faces, theta, knn)
        eigenvalues, eigenvectors = clean_evals_and_evecs(LA.eigh(laplacian))
        if settings.SAVE_MATRICES:
            logger.info("saving binaries...")
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors)
    return eigenvalues, eigenvectors


def integrate_whole_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    function: np.ndarray,
) -> float:
    """
    computes the integral of a function defined on the vertices of the mesh
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function).sum()


def integrate_region_mesh(
    vertices: np.ndarray, faces: np.ndarray, function: np.ndarray, mask: np.ndarray
) -> float:
    """
    computes the integral of a region of a function defines of a mesh vertices
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function * mask).sum()


def clean_evals_and_evecs(
    eigendecomposition: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    need eigenvalues and eigenvectors to be in a certain format
    """
    # access values
    eigenvalues, eigenvectors = eigendecomposition

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].conj().T

    # ensure first element of each eigenvector is positive
    eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)
    return eigenvalues, eigenvectors


def mesh_forward(
    vertices: np.ndarray, faces: np.ndarray, basis_functions: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    computes the mesh forward transform from real space to harmonic space
    """
    u_i = np.zeros(basis_functions.shape[0])
    for i, phi_i in enumerate(basis_functions):
        u_i[i] = integrate_whole_mesh(vertices, faces, u * phi_i.conj())
    return u_i


def mesh_inverse(basis_functions: np.ndarray, u_i: np.ndarray) -> np.ndarray:
    """
    computes the mesh inverse transform from harmonic space to real space
    """
    i_idx = 0
    return (u_i * basis_functions).sum(axis=i_idx)
