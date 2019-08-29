#!/usr/bin/env python
from config import Config
from slepian_functions import SlepianFunctions
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import os


global __location__
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_args() -> Namespace:
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "--num_plots",
        "-p",
        type=valid_plots,
        default=1,
        help="the number of plots to show in descending concentration",
    )
    parser.add_argument(
        "--phi_min",
        "-pmin",
        type=int,
        default=0,
        help="phi minimum in degrees for region - defaults to 0",
    )
    parser.add_argument(
        "--phi_max",
        "-pmax",
        type=int,
        default=360,
        help="phi maximum in degrees for region - defaults to 360",
    )
    parser.add_argument(
        "--theta_min",
        "-tmin",
        type=int,
        default=0,
        help="theta minimum in degrees for region - defaults to 0",
    )
    parser.add_argument(
        "--theta_max",
        "-tmax",
        type=int,
        default=40,
        help="theta maximum in degrees for region - defaults to 40",
    )
    parser.add_argument(
        "--order",
        "-m",
        type=valid_order,
        default=0,
        help="the order interested in if dealing with polar cap",
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
    parser.add_argument(
        "--annotation",
        "-n",
        action="store_false",
        help="flag which if passed removes any annotation",
    )

    args = parser.parse_args()
    return args


def valid_plots(num_plots: int) -> int:
    config = asdict(Config())
    L, num_plots = config["L"], int(num_plots)
    # check if valid range
    if num_plots <= L * L and num_plots > 0:
        return num_plots
    else:
        raise ValueError(f"The number of plots must be positive and less than {L * L}")


def valid_order(order: int) -> int:
    config = asdict(Config())
    L, order = config["L"], int(order)
    # check if valid range
    if abs(order) < L:
        return order
    else:
        raise ValueError(f"The magnitude of the order must and less than {L} ")


if __name__ == "__main__":
    config = asdict(Config())
    args = read_args()

    # if using input from argparse
    config["annotation"] = args.annotation
    config["order"] = args.order
    config["type"] = args.type

    sf = SlepianFunctions(
        args.phi_min, args.phi_max, args.theta_min, args.theta_max, config
    )

    # don't request more plots than exist
    N = args.num_plots
    if N > sf.eigenvectors.shape[0]:
        N = sf.eigenvectors.shape[0]

    # plot the most concentrated up to specified number of plots
    for i in range(N):
        sf.plot(i)
