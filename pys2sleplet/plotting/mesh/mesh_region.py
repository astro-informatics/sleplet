from argparse import ArgumentParser
from pathlib import Path

import cmocean
import numpy as np
import seaborn as sns

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.scripts.plotting_on_mesh import valid_meshes
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.function_dicts import MESHES

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main(mesh_name: str) -> None:
    """
    plots the tiling of the Slepian line
    """
    # initialise mesh and Slepian mesh
    mesh = Mesh(mesh_name, mesh_laplacian=settings.MESH_LAPLACIAN)

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