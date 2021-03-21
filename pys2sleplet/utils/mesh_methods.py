from pathlib import Path

import numpy as np
from igl import cotmatrix, massmatrix, read_triangle_mesh
from scipy.sparse import linalg as LA

from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    return read_triangle_mesh(str(_meshes_path / mesh_name))


def mesh_eigendecomposition(
    vertices: np.ndarray, faces: np.ndarray, number: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of a given mesh according to this example
    https://geometryprocessing.github.io/blackbox-computing-python/geo_viz/#various-examples-eigendecomposition
    """
    logger.info(f"finding {number} basis functions of mesh")
    laplacian = -cotmatrix(vertices, faces)
    eigenvalues, eigenvectors = LA.eigsh(laplacian, number, sigma=0, which="LM")
    return eigenvalues, eigenvectors.T


def mesh_integral(
    vertices: np.ndarray, faces: np.ndarray, function: np.ndarray
) -> float:
    """
    computes the integral of a function defined on the vertices of the mesh
    by multiplying the function by the mass matrix of the mesh
    """
    mass = massmatrix(vertices, faces)
    return mass.dot(function).sum()
