from argparse import ArgumentParser

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sleplet.meshes import Mesh, MeshSlepian
from sleplet.plot_methods import save_plot

sns.set(context="paper")

MESHES = [
    "bird",
    "cheetah",
    "cube",
    "dragon",
    "homer",
    "teapot",
]


def main(mesh_name: str, num_basis_fun: int) -> None:
    """Plots the Slepian eigenvalues of the given mesh."""
    mesh = Mesh(
        mesh_name,
        number_basis_functions=num_basis_fun,
    )
    mesh_slepian = MeshSlepian(mesh)
    plt.semilogx(mesh_slepian.slepian_eigenvalues, "k.")
    plt.axvline(mesh_slepian.N, c="k", ls="--", alpha=0.8)
    plt.annotate(
        f"N={mesh_slepian.N}",
        xy=(mesh_slepian.N, 1),
        xytext=(17, 3),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    ticks = 2 ** np.arange(np.log2(mesh.number_basis_functions) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    save_plot(f"{mesh_name}_slepian_eigenvalues_b{mesh.number_basis_functions}")


if __name__ == "__main__":
    parser = ArgumentParser(description="mesh Slepian eigenvalues")
    parser.add_argument(
        "function",
        type=str,
        choices=MESHES,
        help="mesh to plot",
        default="homer",
        const="homer",
        nargs="?",
    )
    parser.add_argument(
        "--basis",
        "-b",
        type=int,
        help="number of basis functions to extract",
    )
    args = parser.parse_args()
    main(args.function, args.basis)
