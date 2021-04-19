#!/usr/bin/env python
from argparse import ArgumentParser, Namespace

from pys2sleplet.meshes.mesh_plot import MeshPlot
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mesh_methods import MESHES


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    if func_name in MESHES:
        function = func_name
    else:
        raise ValueError("Not a valid function name to plot")
    return function


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create mesh plot")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=MESHES,
        help="mesh to plot",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="index of basis function to plot",
    )
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region (based on settings.toml)",
    )
    parser.add_argument(
        "--slepian",
        "-s",
        action="store_true",
        help="plot the Slepian functions of the region of a mesh",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        default="real",
        const="real",
        choices=["abs", "real", "imag", "sum"],
        help="plotting type: defaults to real",
    )
    return parser.parse_args()


def plot(
    args: Namespace,
    plot_type: str = "real",
    annotations: bool = True,
) -> None:
    """
    master plotting method
    """
    # create mesh plot
    f = MeshPlot(args.function, args.index, slepian=args.slepian)

    # adjust filename
    filename = f.name
    filename += "_slepian" if args.slepian else ""
    filename += f"_rank{args.index}"

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    annotation: list = []

    # do plot
    Plot(
        f.vertices,
        f.faces,
        f.eigenvector,
        filename,
        annotations=annotation,
        plot_type=plot_type,
        region=f.region if args.slepian else None,
    ).execute()


def main() -> None:
    args = read_args()
    plot(args)


if __name__ == "__main__":
    main()
