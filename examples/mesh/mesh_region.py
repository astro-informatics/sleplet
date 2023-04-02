from argparse import ArgumentParser

import cmocean
import numpy as np

from sleplet.meshes import Mesh
from sleplet.plotting import PlotMesh

MESHES = [
    "bird",
    "cheetah",
    "cube",
    "dragon",
    "homer",
    "teapot",
]


def main(mesh_name: str) -> None:
    """Plots the tiling of the Slepian line."""
    # initialise mesh and Slepian mesh
    mesh = Mesh(mesh_name)

    # create region masking
    field = np.zeros(mesh.vertices.shape[0])
    field[mesh.region] = 1

    name = f"{mesh_name}_region"
    PlotMesh(mesh, name, field, colour=cmocean.cm.haline, region=True).execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh tiling")
    parser.add_argument(
        "function",
        type=str,
        choices=MESHES,
        help="mesh to plot",
        default="homer",
        const="homer",
        nargs="?",
    )
    args = parser.parse_args()
    main(args.function)
