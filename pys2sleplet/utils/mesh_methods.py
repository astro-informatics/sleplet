from pathlib import Path
from typing import Tuple

import numpy as np
from igl import cotmatrix, read_triangle_mesh
from scipy.sparse import linalg as LA

from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"


def read_mesh(mesh_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    vertices, triangles = read_triangle_mesh(str(_meshes_path / f"{mesh_name}.obj"))
    return vertices.T, triangles.T


def mesh_eigendecomposition(
    vertices: np.ndarray, triangles: np.ndarray, number: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    computes the eigendecomposition of a given mesh according to this example
    https://geometryprocessing.github.io/blackbox-computing-python/geo_viz/#various-examples-eigendecomposition
    """
    logger.info(f"finding {number} basis functions of mesh")
    laplacian = -cotmatrix(vertices.T, triangles.T)
    eigenvalues, eigenvectors = LA.eigsh(laplacian, number, sigma=0, which="LM")
    return eigenvalues, eigenvectors.T
