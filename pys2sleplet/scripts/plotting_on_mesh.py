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
        "--j_min",
        "-j",
        type=int,
        default=2,
        help="wavelet scale j_min defaults to 2",
    )
    parser.add_argument(
        "--B",
        "-b",
        type=int,
        default=3,
        help="lambda parameter defaults to 3",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        nargs="?",
        default="basis",
        const="basis",
        choices=["basis", "slepian", "wavelets"],
        help="plotting routine: defaults to basis",
    )
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region",
    )
    return parser.parse_args()


def plot(
    args: Namespace,
) -> None:
    """
    master plotting method
    """
    # create mesh plot
    f = MeshPlot(args.function, args.index, args.method, args.B, args.j_min)

    # whether to show region
    show_region = f.region if args.method != "basis" or args.region else None

    # plotly config
    camera_view, colourbar_pos = mesh_plotly_config(args.function)

    # do plot
    Plot(
        f.vertices,
        f.faces,
        f.eigenvector,
        f.name,
        camera_view,
        colourbar_pos,
        region=show_region,
    ).execute()


def main() -> None:
    args = read_args()
    plot(args)


if __name__ == "__main__":
    main()
