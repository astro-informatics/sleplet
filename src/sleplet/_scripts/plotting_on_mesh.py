#!/usr/bin/env python
from argparse import ArgumentParser, Namespace

from sleplet import logger
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils._class_lists import MESH_COEFFICIENTS, MESHES
from sleplet.utils.plot_methods import (
    _coefficients_to_field_mesh,
    compute_amplitude_for_noisy_mesh_plots,
)
from sleplet.utils.string_methods import _convert_classes_list_to_snake_case


def valid_meshes(mesh_name: str) -> str:
    """
    check if valid mesh name
    """
    if mesh_name in MESHES:
        return mesh_name
    else:
        raise ValueError(f"'{mesh_name}' is not a valid mesh name to plot")


def valid_methods(method_name: str) -> str:
    """
    check if valid mesh name
    """
    if method_name in _convert_classes_list_to_snake_case(
        MESH_COEFFICIENTS, word_to_remove="Mesh"
    ):
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
        default="basis_functions",
        const="basis_functions",
        choices=_convert_classes_list_to_snake_case(
            MESH_COEFFICIENTS, word_to_remove="Mesh"
        ),
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


def plot(f: MeshCoefficients, *, normalise: bool, amplitude: float | None) -> None:
    """
    master plotting method
    """
    field = _coefficients_to_field_mesh(f, f.coefficients)
    Plot(
        f.mesh,
        f.name,
        field,
        amplitude=amplitude,
        normalise=normalise,
        region=isinstance(f, MeshSlepianCoefficients),
    ).execute()


def main() -> None:
    args = read_args()
    logger.info(f"mesh: '{args.function}', plotting method: '{args.method}'")

    # function to plot
    mesh = Mesh(args.function, zoom=args.zoom)
    f = MESH_COEFFICIENTS[
        _convert_classes_list_to_snake_case(
            MESH_COEFFICIENTS, word_to_remove="Mesh"
        ).index(args.method)
    ](
        mesh,
        extra_args=args.extra_args,
        noise=args.noise if args.noise is not None else None,
        region=args.region,
    )

    # custom amplitude for noisy plots
    amplitude = compute_amplitude_for_noisy_mesh_plots(f)

    # perform plot
    plot(f, normalise=not args.unnormalise, amplitude=amplitude)


if __name__ == "__main__":
    main()
