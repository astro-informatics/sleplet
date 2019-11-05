#!/usr/bin/env python
from argparse import ArgumentParser, Namespace

import numpy as np
import toml

from pys2sleplet.flm.functions import Functions, functions
from pys2sleplet.plotting.create_plot import CreatePlot
from pys2sleplet.sifting_convolution import SiftingConvolution
from pys2sleplet.utils.plot_methods import calc_resolution
from pys2sleplet.utils.string_methods import filename_angle


def valid_kernels(func_name: str) -> str:
    """
    check if valid kernel
    """
    if func_name in functions():
        return func_name
    else:
        raise ValueError("Not a valid kernel name to convolve")


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    # check if valid function
    if func_name in functions():
        return func_name
    else:
        raise ValueError("Not a valid function name to plot")


def read_args() -> Namespace:
    """
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "flm",
        type=valid_plotting,
        choices=list(functions().keys()),
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
        choices=list(functions().keys()),
        help="glm to perform sifting convolution with i.e. flm x glm*",
    )
    parser.add_argument(
        "--double",
        "-d",
        action="store_true",
        help="flag which if passed creates a double polar cap i.e. polar gap",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=float,
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


def load_config():
    default = toml.load("config.toml")
    args = read_args()
    config = {**default, **args}
    return config


def plot(
    flm_name: str,
    L: int,
    routine: str,
    plot_type: str,
    glm: Functions = None,
    alpha_pi_fraction: float = 0.75,
    beta_pi_fraction: float = 0.125,
    gamma_pi_fraction: float = 0,
) -> None:
    """
    master plotting method
    """
    # setup
    filename = f"{flm_name}_L{L}_"
    flm = Functions(flm_name, L)
    resolution = calc_resolution(L)

    # test for plotting routine
    if routine == "rotate":
        # adjust filename
        filename += f"{routine}_{filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}_"
        # rotate by alpha, beta, gamma
        flm = flm.rotate(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)
    elif routine == "translate":
        # adjust filename - don't add gamma if translation
        filename += f"{routine}_{filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"
        # translate by alpha, beta
        flm = flm.translate(alpha_pi_fraction, beta_pi_fraction)

    if glm is not None:
        # perform convolution
        flm = flm.convolve(glm.flm)
        # adjust filename
        filename += f"convolved_{glm.name}_L{L}_"

    # boost resolution
    flm = flm.boost_res(resolution)

    # add resolution to filename
    filename += f"res{resolution}_"

    # inverse & plot
    f = flm.invert(resolution)

    # check for plotting type
    if plot_type == "real":
        f = f.real
    elif plot_type == "imag":
        f = f.imag
    elif plot_type == "abs":
        f = np.abs(f)
    elif plot_type == "sum":
        f = f.real + f.imag

    # do plot
    filename += plot_type
    cp = CreatePlot(f, resolution, filename, annotations=annotations)
    cp.plotly_plot()


def main():
    # load config
    c = load_config()

    # setup flm
    flm = Functions(c.flm, c.L)
    glm = Functions(c.convolve, c.L)

    plot(c.flm, c.L, c.routine, c.plot_type, c.convolve, c.alpha, c.beta, c.gamma)


if __name__ == "__main__":
    main()
