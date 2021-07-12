from argparse import ArgumentParser

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.scripts.plotting_on_mesh import valid_plotting
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mesh_methods import MESHES


def main(
    mesh_name: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> None:
    # initialise mesh
    mesh = Mesh(mesh_name)

    # create region based on cartesian coordinates
    region_on_vertices = _create_mesh_region(
        mesh.vertices, xmin, xmax, ymin, ymax, zmin, zmax
    )

    # convert regionto faces
    region_on_faces = _convert_vertices_region_to_faces(mesh.faces, region_on_vertices)

    # creates FACES_RANGES
    logger.info(_consecutive_elements(region_on_faces))


def _create_mesh_region(
    vertices: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    """
    creates the boolean region for the given mesh
    """
    return (
        (vertices[:, 0] >= xmin)
        & (vertices[:, 0] <= xmax)
        & (vertices[:, 1] >= ymin)
        & (vertices[:, 1] <= ymax)
        & (vertices[:, 2] >= zmin)
        & (vertices[:, 2] <= zmax)
    )


def _convert_vertices_region_to_faces(
    faces: np.ndarray, region_on_vertices: np.ndarray
) -> np.ndarray:
    """
    converts the region on vertices to faces
    """
    region_reshape = np.argwhere(region_on_vertices).reshape(-1)
    faces_in_region = faces[np.isin(faces, region_reshape)]
    return np.unique(faces_in_region)


def _consecutive_elements(data: np.ndarray, stepsize: int = 1) -> list[list[int]]:
    """
    finds the values of the consecutive elements
    https://stackoverflow.com/a/7353335/7359333
    min/max of these ranges for toml files
    """
    consecutive = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[con.min(), con.max()] for con in consecutive]


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh Slepian eigenvalues")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=MESHES,
        help="mesh to plot",
    )
    parser.add_argument(
        "-xmin",
        type=float,
        default=-np.inf,
        help="minimum x value defaults to negative infinity",
    )
    parser.add_argument(
        "-xmax",
        type=float,
        default=np.inf,
        help="maximum x value defaults to positive infinity",
    )
    parser.add_argument(
        "-ymin",
        type=float,
        default=-np.inf,
        help="minimum y value defaults to negative infinity",
    )
    parser.add_argument(
        "-ymax",
        type=float,
        default=np.inf,
        help="maximum y value defaults to positive infinity",
    )
    parser.add_argument(
        "-zmin",
        type=float,
        default=-np.inf,
        help="minimum z value defaults to negative infinity",
    )
    parser.add_argument(
        "-zmax",
        type=float,
        default=np.inf,
        help="maximum z value defaults to positive infinity",
    )
    args = parser.parse_args()
    main(
        args.function, args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax
    )