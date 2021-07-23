#!/usr/bin/env python
from argparse import ArgumentParser, Namespace

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.function_dicts import MESH_COEFFICIENTS, MESHES
from pys2sleplet.utils.harmonic_methods import mesh_inverse
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_mesh_inverse


def valid_meshes(mesh_name: str) -> str:
    """
    check if valid mesh name
    """
    if mesh_name in MESHES:
        mesh = mesh_name
    else:
        raise ValueError(f"'{mesh_name}' is not a valid mesh name to plot")
    return mesh


def valid_methods(method_name: str) -> str:
    """
    check if valid mesh name
    """
    if method_name in MESH_COEFFICIENTS:
        method = method_name
    else:
        raise ValueError(f"'{method_name}' is not a valid method to plot")
    return method


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create mesh plot")
    parser.add_argument(
        "function",
        type=valid_meshes,
        choices=MESHES,
        help="mesh to plot",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=int,
        nargs="+",
        help="list of extra args for functions",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=valid_methods,
        nargs="?",
        default="basis",
        const="basis",
        choices=MESH_COEFFICIENTS,
        help="plotting routine: defaults to basis",
    )
    parser.add_argument("--noise", "-n", type=int, help="the SNR_IN of the noise level")
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region (based on settings.toml)",
    )
    return parser.parse_args()


def plot(f: MeshCoefficients) -> None:
    """
    master plotting method
    """
    field = (
        mesh_inverse(f.mesh, f.coefficients)
        if isinstance(f, MeshHarmonicCoefficients)
        else slepian_mesh_inverse(f.slepian_mesh, f.coefficients)
    )
    Plot(
        f.mesh, f.name, field, region=not isinstance(f, MeshHarmonicCoefficients)
    ).execute()


def main() -> None:
    args = read_args()
    logger.info(f"mesh: '{args.function}', plotting method: '{args.method}'")

    # function to plot
    mesh = Mesh(args.function, mesh_laplacian=settings.MESH_LAPLACIAN)
    f = MESH_COEFFICIENTS[args.method](
        mesh,
        extra_args=args.extra_args,
        noise=args.noise if args.noise is not None else None,
        region=args.region,
    )

    # perform plot
    plot(f)


if __name__ == "__main__":
    main()
