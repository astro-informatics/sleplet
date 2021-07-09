from argparse import ArgumentParser

import numpy as np
import pandas as pd

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.scripts.plotting_on_mesh import valid_plotting
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
    mesh = Mesh(mesh_name)
    region_on_vertices = _create_mesh_region(
        mesh.vertices, xmin, xmax, ymin, ymax, zmin, zmax
    )
    region_on_faces = _convert_vertices_region_to_faces(mesh.faces, region_on_vertices)
    df = pd.DataFrame(region_on_faces)
    df.to_csv("region.csv", index=False, header=False)


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
        (vertices[:, 0] > xmin)
        & (vertices[:, 0] < xmax)
        & (vertices[:, 1] > ymin)
        & (vertices[:, 1] < ymax)
        & (vertices[:, 2] > zmin)
        & (vertices[:, 2] < zmax)
    )


def _convert_vertices_region_to_faces(
    faces: np.ndarray, region_on_vertices: np.ndarray
) -> np.ndarray:
    """
    converts the region on vertices to faces
    """
    region_reshape = np.argwhere(region_on_vertices).reshape(-1)
    faces_in_region = np.isin(faces, region_reshape).all(axis=1)
    region_on_faces = np.zeros(faces.shape[0])
    region_on_faces[faces_in_region] = 1
    return np.argwhere(region_on_faces)


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
