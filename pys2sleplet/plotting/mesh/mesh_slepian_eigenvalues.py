from argparse import ArgumentParser
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.meshes.slepian_mesh import Mesh, SlepianMesh
from pys2sleplet.scripts.plotting_on_mesh import valid_plotting
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mesh_methods import MESHES
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
    slepian_mesh = SlepianMesh(mesh)
    plt.plot(slepian_mesh.slepian_eigenvalues)
    plt.xlabel("rank")
    plt.ylabel("eigenvalue $\lambda$")
    save_plot(
        fig_path, f"{mesh_name}_slepian_eigenvalues_b{mesh.number_basis_functions}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh Slepian eigenvalues")
    parser.add_argument(
        "function",
        type=valid_plotting,
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
