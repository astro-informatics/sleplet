from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.meshes.classes.mesh_slepian import MeshSlepian
from pys2sleplet.scripts.plotting_on_mesh import valid_meshes
from pys2sleplet.utils.class_lists import MESHES
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.plot_methods import save_plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main(mesh_name: str, num_basis_fun: int) -> None:
    """
    plots the Slepian eigenvalues of the given mesh
    """
    mesh = Mesh(
        mesh_name,
        mesh_laplacian=settings.MESH_LAPLACIAN,
        number_basis_functions=num_basis_fun,
    )
    mesh_slepian = MeshSlepian(mesh)
    plt.semilogx(mesh_slepian.slepian_eigenvalues, "k.")
    ticks = 2 ** np.arange(np.log2(mesh_slepian.N) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    save_plot(
        fig_path, f"{mesh_name}_slepian_eigenvalues_b{mesh.number_basis_functions}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh Slepian eigenvalues")
    parser.add_argument(
        "function",
        type=valid_meshes,
        choices=MESHES,
        help="mesh to plot",
    )
    parser.add_argument(
        "--basis",
        "-b",
        type=int,
        help="number of basis functions to extract",
    )
    args = parser.parse_args()
    main(args.function, args.basis)
