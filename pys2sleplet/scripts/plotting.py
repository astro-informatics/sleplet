#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import List

import numpy as np

from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.functions import function_dict
from pys2sleplet.utils.inputs import config
from pys2sleplet.utils.plot_methods import calc_resolution
from pys2sleplet.utils.string_methods import filename_angle


def valid_kernels(func_name: str) -> str:
    """
    check if valid kernel
    """
    if func_name in function_dict:
        return func_name
    else:
        raise ValueError("Not a valid kernel name to convolve")


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    if func_name in function_dict:
        return func_name
    else:
        raise ValueError("Not a valid function name to plot")


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "flm",
        type=valid_plotting,
        choices=list(function_dict.keys()),
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
        choices=list(function_dict.keys()),
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
    extra_args: List[int],
    plot_type: str,
    routine: str,
    alpha_pi_fraction: float,
    beta_pi_fraction: float,
    gamma_pi_fraction: float,
    g_name: str,
    annotations: List = [],
) -> None:
    """
    master plotting method
    """
    resolution = calc_resolution(L)
    f = function_dict[f_name](L, extra_args)
    filename = f"{f.name}_L{L}_"

    if routine == "rotate":
        filename += f"{routine}_{filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}_"

        # rotate by alpha, beta, gamma
        f.rotate(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)
    elif routine == "translate":
        # don't add gamma if translation
        filename += f"{routine}_{filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"

        # translate by alpha, beta
        f.translate(alpha_pi_fraction, beta_pi_fraction)

    if g_name:
        g = function_dict[g_name](L, extra_args)
        # perform convolution
        f.convolve(g)
        # adjust filename
        filename += f"convolved_{g.name}_L{L}_"

    # inverse & plot
    field = f.field

    # add resolution to filename
    filename += f"res{resolution}_"

    # check for plotting type
    if plot_type == "real":
        field = field.real
    elif plot_type == "imag":
        field = field.imag
    elif plot_type == "abs":
        field = np.abs(field)
    elif plot_type == "sum":
        field = field.real + field.imag

    # do plot
    filename += plot_type
    Plot(field, resolution, filename, annotations).execute()


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
    )


if __name__ == "__main__":
    main()
