import glob
from pathlib import Path
from typing import Optional

import numpy as np
from box import Box
from igl import (
    adjacency_matrix,
    all_pairs_distances,
    average_onto_faces,
    cotmatrix,
    read_triangle_mesh,
    upsample,
)
from numpy import linalg as LA
from scipy.sparse import linalg as LA_sparse

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.integration_methods import integrate_whole_mesh
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import (
    GAUSSIAN_KERNEL_KNN_DEFAULT,
    GAUSSIAN_KERNEL_THETA_DEFAULT,
)

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"
MESHES: set[str] = {
    Path(x.removesuffix(".toml")).stem
    for x in glob.glob(str(_meshes_path / "regions" / "*.toml"))
}


def average_functions_on_vertices_to_faces(
    faces: np.ndarray,
    functions_on_vertices: np.ndarray,
) -> np.ndarray:
    """
    the integrals require all functions to be defined on faces
    this method handles an arbitrary number of functions
    """
    logger.info("converting function on vertices to faces")
    # handle the case of a 1D array
    array_is_1d = len(functions_on_vertices.shape) == 1
    if array_is_1d:
        functions_on_vertices = functions_on_vertices.reshape(1, -1)

    functions_on_faces = np.zeros((functions_on_vertices.shape[0], faces.shape[0]))
    for i, f in enumerate(functions_on_vertices):
        functions_on_faces[i] = average_onto_faces(faces, f)

    # put the vector back in 1D form
    if array_is_1d:
        functions_on_faces = functions_on_faces.reshape(-1)
    return functions_on_faces


def mesh_config(mesh_name: str) -> Box:
    """
    reads in the given mesh region settings file
    """
    return Box.from_toml(filename=_meshes_path / "regions" / f"{mesh_name}.toml")


def mesh_eigendecomposition(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_laplacian: bool = True,
    number_basis_functions: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    computes the eigendecomposition of the mesh represented
    as a graph if already computed then it loads the data
    """
    # determine number of basis functions
    if number_basis_functions is None:
        number_basis_functions = vertices.shape[0] // 4
    logger.info(
        f"finding {number_basis_functions}/{vertices.shape[0]} "
        f"basis functions of {name} mesh"
    )

    # create filenames
    laplacian_type = "mesh" if mesh_laplacian else "graph"
    eigd_loc = (
        _meshes_path
        / "laplacians"
        / laplacian_type
        / "basis_functions"
        / f"{name}_b{number_basis_functions}"
    )
    eval_loc = eigd_loc / "eigenvalues.npy"
    evec_loc = eigd_loc / "eigenvectors.npy"

    if eval_loc.exists() and evec_loc.exists():
        logger.info("binaries found - loading...")
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
    else:
        if laplacian_type == "mesh":
            laplacian = _mesh_laplacian(vertices, faces)
            eigenvalues, eigenvectors = LA_sparse.eigsh(
                laplacian, number_basis_functions, which="LM", sigma=0
            )
        else:
            laplacian = _graph_laplacian(
                vertices,
                faces,
            )
            eigenvalues, eigenvectors = LA.eigh(laplacian)
        eigenvectors = _orthonormalise_basis_functions(eigenvectors.T)
        if settings.SAVE_MATRICES:
            logger.info("saving binaries...")
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors)
    return eigenvalues, eigenvectors, number_basis_functions


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    data = mesh_config(mesh_name)
    vertices, faces = read_triangle_mesh(str(_meshes_path / "polygons" / data.FILENAME))
    return upsample(vertices, faces, number_of_subdivs=data.UPSAMPLE)


def _graph_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    theta: float = GAUSSIAN_KERNEL_THETA_DEFAULT,
    knn: int = GAUSSIAN_KERNEL_KNN_DEFAULT,
) -> np.ndarray:
    """
    computes the graph laplacian L = D - W where D
    is the degree matrix and W is the weighting function
    """
    rows = 0
    A = adjacency_matrix(faces)
    D = np.diagflat(A.sum(axis=rows))
    W = _weighting_function(D, A, vertices, theta, knn)
    return np.asarray(D - W)


def _mesh_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    computes the cotagent mesh laplacian
    """
    return -cotmatrix(vertices, faces)


def _orthonormalise_basis_functions(basis_functions: np.ndarray) -> np.ndarray:
    """
    for computing the Slepian D matrix the basis functions must be orthonormal
    """
    logger.info("orthonormalising basis functions")
    factor = np.zeros(basis_functions.shape[0])
    for i, phi_i in enumerate(basis_functions):
        factor[i] = integrate_whole_mesh(phi_i, phi_i)
    normalisation = np.sqrt(factor).reshape(-1, 1)
    return basis_functions / normalisation


def _weighting_function(
    D: np.ndarray, A: np.ndarray, vertices: np.ndarray, theta: float, knn: int
) -> np.ndarray:
    """
    thresholded Gaussian kernel weighting function
    """
    W = np.exp(
        -all_pairs_distances(vertices, vertices, squared=True) / (2 * theta ** 2)
    )
    knn_threshold = D <= knn
    return W * A * knn_threshold
