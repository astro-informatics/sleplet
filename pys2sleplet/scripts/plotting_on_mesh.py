#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.utils.class_lists import MESH_COEFFICIENTS, MESHES
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import mesh_inverse
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_mesh_inverse
from pys2sleplet.utils.string_methods import convert_classes_list_to_snake_case


def valid_meshes(mesh_name: str) -> str:
    """
    check if valid mesh name
    """
    if mesh_name in convert_classes_list_to_snake_case(MESHES):
        return mesh_name
    else:
        raise ValueError(f"'{mesh_name}' is not a valid mesh name to plot")


def valid_methods(method_name: str) -> str:
    """
    check if valid mesh name
    """
    if method_name in convert_classes_list_to_snake_case(MESH_COEFFICIENTS):
        return method_name
    else:
        raise ValueError(f"'{method_name}' is not a valid method to plot")


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
        choices=convert_classes_list_to_snake_case(MESH_COEFFICIENTS),
        help="plotting routine: defaults to basis",
    )
    parser.add_argument("--noise", "-n", type=int, help="the SNR_IN of the noise level")
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region",
    )
    parser.add_argument(
        "--unnormalise",
        "-u",
        action="store_true",
        help="flag turns off normalisation for plot",
    )
    parser.add_argument(
        "--zoom",
        "-z",
        action="store_true",
        help="flag which zooms in on the region of interest",
    )
    return parser.parse_args()


def plot(f: MeshCoefficients, normalise: bool, amplitude: Optional[float]) -> None:
    """
    master plotting method
    """
    field = _coefficients_to_field(f, f.coefficients)
    Plot(
        f.mesh,
        f.name,
        field,
        amplitude=amplitude,
        normalise=normalise,
        region=not isinstance(f, MeshHarmonicCoefficients),
    ).execute()


def _coefficients_to_field(f: MeshCoefficients, coefficients: np.ndarray) -> np.ndarray:
    """
    computes the field over the whole mesh from the harmonic/Slepian coefficients
    """
    return (
        mesh_inverse(f.mesh, coefficients)
        if isinstance(f, MeshHarmonicCoefficients)
        else slepian_mesh_inverse(f.slepian_mesh, coefficients)
    )


def _compute_amplitude_for_noisy_plots(f: MeshCoefficients) -> Optional[float]:
    """
    for the noised plots fix the amplitude to the initial data
    """
    return (
        np.abs(_coefficients_to_field(f, f.unnoised_coefficients)).max()
        if f.noise is not None
        else None
    )


def main() -> None:
    args = read_args()
    logger.info(f"mesh: '{args.function}', plotting method: '{args.method}'")

    # function to plot
    mesh = Mesh(args.function, mesh_laplacian=settings.MESH_LAPLACIAN, zoom=args.zoom)
    f = MESH_COEFFICIENTS[
        convert_classes_list_to_snake_case(MESH_COEFFICIENTS).index(args.method)
    ](
        mesh,
        extra_args=args.extra_args,
        noise=args.noise if args.noise is not None else None,
        region=args.region,
    )

    # custom amplitude for noisy plots
    amplitude = _compute_amplitude_for_noisy_plots(f)

    # perform plot
    plot(f, not args.unnormalise, amplitude)


if __name__ == "__main__":
    main()
