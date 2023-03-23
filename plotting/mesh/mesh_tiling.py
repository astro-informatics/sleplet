from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.classes.mesh_slepian import MeshSlepian
from sleplet.scripts.plotting_on_mesh import valid_meshes
from sleplet.utils.class_lists import MESHES
from sleplet.utils.plot_methods import save_plot
from sleplet.utils.wavelet_methods import create_kappas

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

B = 3
J_MIN = 2
STEP = 0.01


def main(mesh_name: str) -> None:
    """
    plots the tiling of the Slepian line
    """
    # initialise mesh and Slepian mesh
    mesh = Mesh(mesh_name)
    mesh_slepian = MeshSlepian(mesh)

    # set up x-axis
    xlim = mesh.mesh_eigenvalues.shape[0]
    x = np.arange(xlim)

    # scaling function
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappas = create_kappas(xlim, B, J_MIN)
    yi = pchip(x, kappas[0])
    plt.semilogx(xi, yi(xi), label=r"$\Phi_p$")

    # wavelets
    for j, k in enumerate(kappas[1:]):
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{{{j+J_MIN}}}_p$")

    # add vertical line
    plt.axvline(mesh_slepian.N, color="k", linestyle="dashed")
    plt.annotate(
        f"N={mesh_slepian.N}",
        xy=(mesh_slepian.N, 1),
        xytext=(17, 3),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )

    # format plot
    plt.xlim(1, xlim)
    ticks = 2 ** np.arange(np.log2(xlim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.legend()
    save_plot(fig_path, f"{mesh_name}_slepian_tiling_b{mesh.mesh_eigenvalues.shape[0]}")


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
