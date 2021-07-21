from argparse import ArgumentParser

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_field import SlepianMeshField
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_wavelets import (
    SlepianMeshWavelets,
)
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.scripts.plotting_on_mesh import valid_plotting
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.denoising import denoising_mesh_slepian
from pys2sleplet.utils.function_dicts import MESHES
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_mesh_inverse

B = 3
J_MIN = 2
N_SIGMA = 2
SNR_IN = -10


def main(mesh_name: str, snr: int, sigma: int) -> None:
    """
    denoising demo using Slepian wavelets
    """
    logger.info(f"SNR={snr}, n_sigma={sigma}")
    # setup
    mesh = Mesh(mesh_name, mesh_laplacian=settings.MESH_LAPLACIAN)

    # create map & noised map
    fun = SlepianMeshField(mesh)
    fun_noised = SlepianMeshField(mesh, noise=snr)

    # create wavelets
    smw = SlepianMeshWavelets(mesh, B=B, j_min=J_MIN)

    # fix amplitude
    amplitude = slepian_mesh_inverse(fun_noised.slepian_mesh, fun_noised.coefficients)

    f = denoising_mesh_slepian(fun, fun_noised, smw, snr, sigma)
    name = f"{mesh_name}_snr{snr}_n{sigma}_denoised"
    Plot(mesh, name, f, amplitude=amplitude, region=True).execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="denoising")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=MESHES,
        help="mesh to plot",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=int,
        default=SNR_IN,
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=int,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.function, args.noise, args.sigma)
