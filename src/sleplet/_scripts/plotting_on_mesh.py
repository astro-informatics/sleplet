import argparse
import logging

import sleplet._class_lists
import sleplet._string_methods
import sleplet.meshes.mesh
import sleplet.meshes.mesh_coefficients
import sleplet.plot_methods
import sleplet.plotting._create_plot_mesh

_logger = logging.getLogger(__name__)


def valid_meshes(mesh_name: str) -> str:
    """Check if valid mesh name."""
    if mesh_name in sleplet._class_lists.MESHES:
        return mesh_name
    msg = f"'{mesh_name}' is not a valid mesh name to plot"
    raise ValueError(msg)


def valid_methods(method_name: str) -> str:
    """Check if valid mesh name."""
    if method_name in sleplet._string_methods.convert_classes_list_to_snake_case(
        sleplet._class_lists.MESH_COEFFICIENTS,
        word_to_remove="Mesh",
    ):
        return method_name
    msg = f"'{method_name}' is not a valid method to plot"
    raise ValueError(msg)


def read_args() -> argparse.Namespace:
    """Read args from the command line."""
    parser = argparse.ArgumentParser(description="Create mesh plot")
    parser.add_argument(
        "function",
        type=valid_meshes,
        choices=sleplet._class_lists.MESHES,
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
        choices=sleplet._string_methods.convert_classes_list_to_snake_case(
            sleplet._class_lists.MESH_COEFFICIENTS,
            word_to_remove="Mesh",
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
        "--version",
        "-v",
        action="version",
        version=sleplet.__version__,
    )
    parser.add_argument(
        "--zoom",
        "-z",
        action="store_true",
        help="flag which zooms in on the region of interest",
    )
    return parser.parse_args()


def plot(
    f: sleplet.meshes.mesh_coefficients.MeshCoefficients,
    *,
    normalise: bool,
    amplitude: float | None,
) -> None:
    """Master plotting method."""
    field = sleplet.plot_methods._coefficients_to_field_mesh(f, f.coefficients)
    sleplet.plotting._create_plot_mesh.PlotMesh(
        f.mesh,
        f.name,
        field,
        amplitude=amplitude,
        normalise=normalise,
        region=hasattr(f, "mesh_slepian"),
    ).execute()


def main() -> None:
    args = read_args()
    msg = f"mesh: '{args.function}', plotting method: '{args.method}'"
    _logger.info(msg)

    # function to plot
    mesh = sleplet.meshes.mesh.Mesh(args.function, zoom=args.zoom)
    f = sleplet._class_lists.MESH_COEFFICIENTS[
        sleplet._string_methods.convert_classes_list_to_snake_case(
            sleplet._class_lists.MESH_COEFFICIENTS,
            word_to_remove="Mesh",
        ).index(args.method)
    ](
        mesh,
        extra_args=args.extra_args,
        noise=args.noise if args.noise is not None else None,
        region=args.region,
    )

    # custom amplitude for noisy plots
    amplitude = sleplet.plot_methods.compute_amplitude_for_noisy_mesh_plots(f)

    # perform plot
    plot(f, normalise=not args.unnormalise, amplitude=amplitude)


if __name__ == "__main__":
    main()
