import glob
from pathlib import Path
from typing import Union

import numpy as np
from box import Box
from igl import (
    adjacency_matrix,
    all_pairs_distances,
    cotmatrix,
    massmatrix,
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
    return (
        (vertices[:, 0] > data.XMIN)
        & (vertices[:, 0] < data.XMAX)
        & (vertices[:, 1] > data.YMIN)
        & (vertices[:, 1] < data.YMAX)
        & (vertices[:, 2] > data.ZMIN)
        & (vertices[:, 2] < data.ZMAX)
    )


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
    computes the graph laplacian L = D - W
    where D is the degree matrix and W is the weighting function
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
    name: str, vertices: np.ndarray, faces: np.ndarray, laplacian_type: str = "mesh"
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of the mesh represented as a graph
    if already computed then it loads the data
    """
    logger.info(f"finding {vertices.shape[0]} basis functions of {name} mesh")

    # read in polygon data
    data = _read_toml(name)

    # create filenames
    eigd_loc = _meshes_path / "basis_functions" / name / f"{laplacian_type}_laplacian"
    eval_loc = eigd_loc / "eigenvalues.npy"
    evec_loc = eigd_loc / "eigenvectors.npy"

    if eval_loc.exists() and evec_loc.exists():
        logger.info("binaries found - loading...")
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
    else:
        if laplacian_type == "mesh":
            laplacian = _mesh_laplacian(vertices, faces)
            eigendecomposition = LA_sparse.eigsh(
                laplacian, data.NUMBER, which="LM", sigma=0
            )
        else:
            laplacian = _graph_laplacian(
                vertices, faces, theta=data.THETA, knn=data.KNN
            )
            eigendecomposition = LA.eigh(laplacian)
        eigenvalues, eigenvectors = clean_evals_and_evecs(eigendecomposition)
        if settings.SAVE_MATRICES:
            logger.info("saving binaries...")
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors)
    return eigenvalues, eigenvectors


def integrate_whole_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    function: Union[np.ndarray, int],
) -> float:
    """
    computes the integral of a function defined on the vertices
    of the mesh or the same constant value at each vertex
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function).sum()


def integrate_region_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    function: Union[np.ndarray, int],
    mask: np.ndarray,
) -> float:
    """
    computes the integral of a region of a function defines on the
    vertices of the mesh or the same constant value at each vertex
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
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T

    # ensure first element of each eigenvector is positive
    eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]
    return eigenvalues, eigenvectors


def mesh_forward(
    vertices: np.ndarray, faces: np.ndarray, basis_functions: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    computes the mesh forward transform from real space to harmonic space
    """
    u_i = np.zeros(basis_functions.shape[0])
    for i, phi_i in enumerate(basis_functions):
        u_i[i] = integrate_whole_mesh(vertices, faces, u * phi_i)
    return u_i


def mesh_inverse(basis_functions: np.ndarray, u_i: np.ndarray) -> np.ndarray:
    """
    computes the mesh inverse transform from harmonic space to real space
    """
    i_idx = 0
    return (u_i[:, np.newaxis] * basis_functions).sum(axis=i_idx)


def compute_shannon(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
) -> int:
    """
    computes the effective Shannon number for a region of a mesh
    """
    function_value = 1
    num_basis_fun = vertices.shape[0]
    region_area = integrate_region_mesh(vertices, faces, function_value, mask)
    mesh_area = integrate_whole_mesh(vertices, faces, function_value)
    return round(region_area / mesh_area * num_basis_fun)
