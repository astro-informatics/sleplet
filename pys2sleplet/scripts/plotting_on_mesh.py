#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.plotting.create_plot_mesh import Plot
from pys2sleplet.utils.function_dicts import MESHES
from pys2sleplet.utils.logger import logger


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
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=list(MESHES.keys()),
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
    f: Mesh,
    plot_type: str = "real",
    annotations: bool = True,
) -> None:
    """
    master plotting method
    """
    filename = Path(f.name).stem

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    annotation: List = []

    colour = np.ones(f.eigenvectors[f.number].shape)
    colour[np.argwhere(f.region)] = 0

    # do plot
    Plot(
        f.vertices,
        f.triangles,
        colour,
        filename,
        annotations=annotation,
        plot_type=plot_type,
    ).execute()


def main() -> None:
    args = read_args()

    f = MESHES[args.function](
        extra_args=args.extra_args,
    )

    plot(
        f,
        plot_type=args.type,
    )


if __name__ == "__main__":
    main()
