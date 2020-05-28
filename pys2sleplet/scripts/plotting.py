#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np

from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import config
from pys2sleplet.utils.function_dicts import FUNCTIONS
from pys2sleplet.utils.string_methods import filename_angle


def valid_kernels(func_name: str) -> str:
    """
    check if valid kernel
    """
    if func_name in FUNCTIONS:
        function = func_name
    else:
        raise ValueError("Not a valid kernel name to convolve")
    return function


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    if func_name in FUNCTIONS:
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
        "flm",
        type=valid_plotting,
        choices=list(FUNCTIONS.keys()),
        help="flm to plot on the sphere",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.75,
        help="alpha/phi pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--annotation",
        "-n",
        action="store_false",
        help="flag which if passed removes any annotation",
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=0.125,
        help="beta/theta pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--convolve",
        "-c",
        type=valid_kernels,
        default=None,
        choices=list(FUNCTIONS.keys()),
        help="glm to perform sifting convolution with i.e. flm x glm*",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=int,
        nargs="+",
        help="list of extra args for functions",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=float,
        default=0,
        help="gamma pi fraction - defaults to 0 - rotation only",
    )
    parser.add_argument(
        "--routine",
        "-r",
        type=str,
        nargs="?",
        default="north",
        const="north",
        choices=["north", "rotate", "translate"],
        help="plotting routine: defaults to north",
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
    args = parser.parse_args()
    return args


def plot(
    f_name: str,
    L: int,
    extra_args: Optional[List[int]],
    plot_type: str,
    routine: str,
    alpha_pi_fraction: float,
    beta_pi_fraction: float,
    gamma_pi_fraction: float,
    g_name: Optional[str],
    annotations: bool,
) -> None:
    """
    master plotting method
    """
    f = FUNCTIONS[f_name](L, extra_args)
    filename = f"{f.name}_L{L}_"

    if routine == "rotate":
        filename += (
            f"{routine}_"
            f"{filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}_"
        )

        # rotate by alpha, beta, gamma
        f.rotate(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)
    elif routine == "translate":
        # don't add gamma if translation
        filename += f"{routine}_{filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"

        # translate by alpha, beta
        f.translate(alpha_pi_fraction, beta_pi_fraction)

    if g_name:
        g = FUNCTIONS[g_name](L, extra_args)
        # perform convolution
        f.convolve(g.multipole)
        # adjust filename
        filename += f"convolved_{g.name}_L{L}_"

    # add resolution to filename
    filename += f"res{f.resolution}_"

    # check for plotting type
    if plot_type == "real":
        field = f.field_padded.real
    elif plot_type == "imag":
        field = f.field_padded.imag
    elif plot_type == "abs":
        field = np.abs(f.field_padded)
    elif plot_type == "sum":
        field = f.field_padded.real + f.field_padded.imag

    # turn off annotation if needed
    if annotations:
        annotation = f.annotations
    else:
        annotation = []

    # do plot
    filename += plot_type
    Plot(field, f.resolution, filename, annotation).execute()


def main() -> None:
    args = read_args()

    plot(
        args.flm,
        config.L,
        args.extra_args,
        args.type,
        args.routine,
        args.alpha,
        args.beta,
        args.gamma,
        args.convolve,
        args.annotation,
    )


if __name__ == "__main__":
    main()
