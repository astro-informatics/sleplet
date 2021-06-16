#!/usr/bin/env python
from argparse import ArgumentParser, Namespace

from pys2sleplet.meshes.mesh_plot import MeshPlot
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.utils.mesh_methods import MESHES, mesh_plotly_config


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
        help="flag which masks the function for a region",
    )
    parser.add_argument(
        "--slepian",
        "-s",
        action="store_true",
        help="plot the Slepian functions of the region of a mesh",
    )
    return parser.parse_args()


def plot(
    args: Namespace,
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

    # whether to show region
    show_region = f.region if args.slepian or args.region else None

    # plotly config
    camera_view, colourbar_pos = mesh_plotly_config(args.function)

    # do plot
    Plot(
        f.vertices,
        f.faces,
        f.eigenvector,
        filename,
        camera_view,
        colourbar_pos,
        region=show_region,
    ).execute()


def main() -> None:
    args = read_args()
    plot(args)


if __name__ == "__main__":
    main()
