from pathlib import Path

import numpy as np
from box import Box
from igl import cotmatrix, massmatrix, read_triangle_mesh
from scipy.sparse import linalg as LA

from pys2sleplet.utils.logger import logger

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"


def _read_toml(mesh_name: str) -> Box:
    """
    reads in the given mesh region settings file
    """
    return Box.from_toml(filename=_meshes_path / "region" / f"{mesh_name}.toml")


def read_mesh(mesh_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    reads in the given mesh
    """
    data = _read_toml(mesh_name)
    return read_triangle_mesh(str(_meshes_path / "polygons" / data.FILENAME))


def create_mesh_region(mesh_name: str, vertices: np.ndarray) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    data = _read_toml(mesh_name)
    return (
        (vertices[:, 0] > data.XMIN)
        & (vertices[:, 0] < data.XMAX)
        & (vertices[:, 1] > data.XMIN)
        & (vertices[:, 1] < data.XMAX)
        & (vertices[:, 2] > data.XMIN)
        & (vertices[:, 2] < data.XMAX)
    )


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
    mass = massmatrix(vertices * mask, faces)
    return mass.dot(function).sum()
