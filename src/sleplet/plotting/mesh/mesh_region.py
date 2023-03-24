from argparse import ArgumentParser
from pathlib import Path

import cmocean
import numpy as np

from sleplet.meshes.classes.mesh import Mesh
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.scripts.plotting_on_mesh import valid_meshes
from sleplet.utils.class_lists import MESHES

fig_path = Path(__file__).resolve().parents[2] / "figures"


def main(mesh_name: str) -> None:
    """
    plots the tiling of the Slepian line
    """
    # initialise mesh and Slepian mesh
    mesh = Mesh(mesh_name)

    # create region masking
    field = np.zeros(mesh.vertices.shape[0])
    field[mesh.region] = 1

    name = f"{mesh_name}_region"
    Plot(mesh, name, field, colour=cmocean.cm.haline, region=True).execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh tiling")
    parser.add_argument(
        "function",
        type=valid_meshes,
        choices=MESHES,
        help="mesh to plot",
    )
    args = parser.parse_args()
    main(args.function)
