from pathlib import Path

import numpy as np
from box import Box
from igl import average_onto_faces, cotmatrix, read_triangle_mesh, upsample
from scipy.sparse import linalg as LA_sparse

from sleplet.utils.config import settings
from sleplet.utils.integration_methods import integrate_whole_mesh
from sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"


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


def create_mesh_region(mesh_config: Box, vertices: np.ndarray) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    return (
        (vertices[:, 0] >= mesh_config.XMIN)
        & (vertices[:, 0] <= mesh_config.XMAX)
        & (vertices[:, 1] >= mesh_config.YMIN)
        & (vertices[:, 1] <= mesh_config.YMAX)
        & (vertices[:, 2] >= mesh_config.ZMIN)
        & (vertices[:, 2] <= mesh_config.ZMAX)
    )


def extract_mesh_config(mesh_name: str) -> Box:
    """
    reads in the given mesh region settings file
    """
    return Box.from_toml(filename=_meshes_path / "regions" / f"{mesh_name}.toml")


def mesh_eigendecomposition(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    number_basis_functions: int | None = None,
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
    eigd_loc = (
        _meshes_path
        / "laplacians"
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
        laplacian = _mesh_laplacian(vertices, faces)
        eigenvalues, eigenvectors = LA_sparse.eigsh(
            laplacian, k=number_basis_functions, which="LM", sigma=0
        )
        eigenvectors = _orthonormalise_basis_functions(vertices, faces, eigenvectors.T)
        if settings.SAVE_MATRICES:
            logger.info("saving binaries...")
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors)
    return eigenvalues, eigenvectors, number_basis_functions


def read_mesh(mesh_config: Box) -> tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    vertices, faces = read_triangle_mesh(
        str(_meshes_path / "polygons" / mesh_config.FILENAME)
    )
    return upsample(vertices, faces, number_of_subdivs=mesh_config.UPSAMPLE)


def _mesh_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    computes the cotagent mesh laplacian
    """
    return -cotmatrix(vertices, faces)


def _orthonormalise_basis_functions(
    vertices: np.ndarray, faces: np.ndarray, basis_functions: np.ndarray
) -> np.ndarray:
    """
    for computing the Slepian D matrix the basis functions must be orthonormal
    """
    logger.info("orthonormalising basis functions")
    factor = np.zeros(basis_functions.shape[0])
    for i, phi_i in enumerate(basis_functions):
        factor[i] = integrate_whole_mesh(vertices, faces, phi_i, phi_i)
    normalisation = np.sqrt(factor).reshape(-1, 1)
    return basis_functions / normalisation
