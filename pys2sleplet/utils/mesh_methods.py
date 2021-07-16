import glob
from functools import reduce
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
)
from numpy import linalg as LA
from plotly.graph_objs.layout.scene import Camera
from scipy.sparse import linalg as LA_sparse

from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plotly_methods import create_camera
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
    vertices_selected = np.zeros(vertices.shape[0], dtype=int)
    for vertices_min, vertices_max in data.VERTICES_RANGES.to_list():
        vertices_selected[vertices_min : vertices_max + 1] = 1
    return vertices_selected


def mesh_plotly_config(mesh_name: str) -> tuple[Camera, float]:
    """
    creates plotly camera view for a given mesh
    """
    data = _read_toml(mesh_name)
    return (
        create_camera(data.CAMERA_X, data.CAMERA_Y, data.CAMERA_Z, data.ZOOM),
        data.COLOURBAR_POS,
    )


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


def mesh_eigendecomposition(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_laplacian: bool = True,
    number_basis_functions: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of the mesh represented
    as a graph if already computed then it loads the data
    """
    # read in polygon data
    data = _read_toml(name)

    # determine number of basis functions
    if number_basis_functions is None:
        number_basis_functions = data.NUMBER
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
                vertices, faces, theta=data.THETA, knn=data.KNN
            )
            eigenvalues, eigenvectors = LA.eigh(laplacian)
        eigenvectors = _orthonormalise_basis_functions(eigenvectors.T)
        if settings.SAVE_MATRICES:
            logger.info("saving binaries...")
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors)
    return eigenvalues, eigenvectors


def integrate_whole_mesh(*functions: np.ndarray) -> float:
    """
    computes the integral of functions on the vertices
    """
    multiplied_inputs = _multiply_args(*functions)
    return multiplied_inputs.sum()


def integrate_region_mesh(
    mask: np.ndarray,
    *functions: np.ndarray,
) -> float:
    """
    computes the integral of a region of functions on the vertices
    """
    multiplied_inputs = _multiply_args(*functions)
    return (multiplied_inputs * mask).sum()


def _multiply_args(*args: np.ndarray) -> np.ndarray:
    """
    method to multiply an unknown number of arguments
    """
    return reduce((lambda x, y: x * y), args)


def mesh_forward(basis_functions: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    computes the mesh forward transform from real space to harmonic space
    """
    u_i = np.zeros(basis_functions.shape[0])
    for i, phi_i in enumerate(basis_functions):
        u_i[i] = integrate_whole_mesh(u, phi_i)
    return u_i


def mesh_inverse(basis_functions: np.ndarray, u_i: np.ndarray) -> np.ndarray:
    """
    computes the mesh inverse transform from harmonic space to real space
    """
    i_idx = 0
    return (u_i[:, np.newaxis] * basis_functions).sum(axis=i_idx)


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


def bandlimit_signal(
    vertices: np.ndarray, faces: np.ndarray, basis_functions: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    ensures that signal in pixel space is bandlimited
    """
    u_i = mesh_forward(
        basis_functions,
        u,
    )
    return mesh_inverse(basis_functions, u_i)


def convert_region_on_vertices_to_faces(
    faces: np.ndarray, region_on_vertices: np.ndarray
) -> np.ndarray:
    """
    converts the region on vertices to faces
    """
    region_reshape = np.argwhere(region_on_vertices).reshape(-1)
    faces_in_region = np.isin(faces, region_reshape).all(axis=1)
    region_on_faces = np.zeros(faces.shape[0])
    region_on_faces[faces_in_region] = 1
    return region_on_faces
