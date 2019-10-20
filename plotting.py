#!/usr/bin/env python
from config import Config
from pys2sleplet.sifting_convolution import SiftingConvolution
from pys2sleplet.slepian_functions import SlepianFunctions

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from fractions import Fraction
import numpy as np
import os
import pyssht as ssht
import scipy.io as sio
from typing import List, Tuple


global __location__
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_angle_num_dem(angle_fraction: float) -> Tuple[int, int]:
    """
    """
    angle = Fraction(angle_fraction).limit_denominator()
    return angle.numerator, angle.denominator


def filename_args(angle: float, arg_name: str) -> str:
    """
    """
    filename = "_"
    num, dem = get_angle_num_dem(angle)
    filename += f"{num}{arg_name}"
    if angle < 1 and angle != 0:
        filename += f"{dem}"
    return filename


def read_args() -> Namespace:
    """
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "flm",
        type=valid_plotting,
        choices=list(functions.keys()),
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
        choices=list(functions.keys()),
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


# -----------------------------------------
# ---------- convolution kernels ----------
# -----------------------------------------


def dirac_delta() -> Tuple[np.ndarray, str, dict]:
    """
    """
    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L = config["L"]

    # filename
    func_name = "dirac_delta"

    # create flm
    flm = np.zeros((L * L), dtype=complex)
    for ell in range(L):
        ind = ssht.elm2ind(ell, m=0)
        flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))

    return flm, func_name, config


def elongated_gaussian(args: List[int] = [0.0, -3.0]) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # args
    try:
        t_sig, p_sig = args.pop(0), args.pop(0)
    except IndexError:
        raise ValueError("function requires exactly two extra args")

    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L, reality = config["L"], config["reality"]

    # validation
    if not t_sig.is_integer():
        raise ValueError("theta sigma should be an integer")
    if not p_sig.is_integer():
        raise ValueError("phi sigma should be an integer")
    t_sig, p_sig = 10 ** t_sig, 10 ** p_sig

    # filename
    func_name = (
        f"elongated_gaussian{filename_args(t_sig, 'tsig')}"
        f"{filename_args(p_sig, 'psig')}"
    )

    def grid_fun(
        theta: np.ndarray,
        phi: np.ndarray,
        theta_0: float = 0,
        phi_0: float = np.pi,
        theta_sig: float = t_sig,
        phi_sig: float = p_sig,
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(
            -(((theta - theta_0) / theta_sig) ** 2 + ((phi - phi_0) / phi_sig) ** 2) / 2
        )
        return f

    thetas, phis = ssht.sample_positions(L, Grid=True, Method="MWSS")
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, L, Reality=reality, Method="MWSS")

    return flm, func_name, config


def gaussian(args: List[int] = [3.0]) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L = config["L"]

    # validation
    if not args[0].is_integer():
        raise ValueError("sigma should be an integer")
    sig = 10 ** args[0]

    # filename
    func_name = f"gaussian{filename_args(sig, 'sig')}"

    # create flm
    flm = np.zeros((L * L), dtype=complex)
    for ell in range(L):
        ind = ssht.elm2ind(ell, m=0)
        flm[ind] = np.exp(-ell * (ell + 1) / (2 * sig * sig))

    return flm, func_name, config


def harmonic_gaussian(args: List[float] = [3.0, 3.0]) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # args
    try:
        l_sig, m_sig = args.pop(0), args.pop(0)
    except IndexError:
        raise ValueError("function requires exactly two extra args")

    # setup
    config = asdict(Config())
    extra = dict(reality=False)
    config = {**config, **extra}
    L = config["L"]

    # validation
    if not l_sig.is_integer():
        raise ValueError("l sigma should be an integer")
    if not m_sig.is_integer():
        raise ValueError("m sigma should be an integer")
    l_sig, m_sig = 10 ** l_sig, 10 ** m_sig

    # filename
    func_name = (
        f"harmonic_gaussian{filename_args(l_sig, 'lsig')}{filename_args(m_sig, 'msig')}"
    )

    # create flm
    flm = np.zeros((L * L), dtype=complex)
    for ell in range(L):
        upsilon_l = np.exp(-((ell / l_sig) ** 2) / 2)
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = upsilon_l * np.exp(-((m / m_sig) ** 2) / 2)

    return flm, func_name, config


def identity() -> Tuple[np.ndarray, str, dict]:
    """
    """
    # setup
    config = asdict(Config())
    extra = dict(reality=False)
    config = {**config, **extra}
    L = config["L"]

    # filename
    func_name = "identity"

    # create flm
    flm = np.ones((L * L)) + 1j * np.zeros((L * L))

    return flm, func_name, config


def slepian(
    args: List[int] = [0.0, 360.0, 0.0, 40.0, 0.0, 0.0, 0.0]
) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # args
    try:
        phi_min, phi_max, theta_min, theta_max = (
            args.pop(0),
            args.pop(0),
            args.pop(0),
            args.pop(0),
        )
    except IndexError:
        raise ValueError("function requires at least four extra args")
    try:
        rank = args.pop(0)
    except IndexError:
        rank = 0.0  # the most concentrated Slepian rank
    try:
        order = args.pop(0)
    except IndexError:
        order = 0.0  # D matrix corresponding to m=0 for polar cap
    try:
        double = args.pop(0)
    except IndexError:
        double = 0.0  # set boolean switch for polar gap off

    # setup
    config = asdict(Config())
    extra = dict(reality=False)
    config = {**config, **extra}
    L, ncpu, save_matrices = config["L"], config["ncpu"], config["save_matrices"]

    # initialise class
    sf = SlepianFunctions(
        L, phi_min, phi_max, theta_min, theta_max, order, double, ncpu, save_matrices
    )

    # validation
    if not rank.is_integer() or rank < 0:
        raise ValueError(f"Slepian concentration rank should be a positive integer")
    if sf.is_polar_cap:
        if rank >= L - abs(sf.order):
            raise ValueError(
                f"Slepian concentration rank should be less than {L - abs(sf.order)}"
            )
    else:
        if rank >= L * L:
            raise ValueError(f"Slepian concentration rank should be less than {L * L}")

    # filename
    rank = int(rank)
    func_name = f"slepian{sf.filename_angle()}{sf.filename}_rank{rank}"

    # create flm
    flm = sf.eigenvectors[rank]
    print(f"Eigenvalue {rank}: {sf.eigenvalues[rank]:e}")

    # annotation
    config["annotations"] = sf.annotations()

    return flm, func_name, config


def spherical_harmonic(args: List[int] = [0.0, 0.0]) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # args
    try:
        ell, m = args.pop(0), args.pop(0)
    except IndexError:
        raise ValueError("function requires exactly two extra args")

    # setup
    config = asdict(Config())
    extra = dict(reality=False)
    config = {**config, **extra}
    L = config["L"]

    # validation
    if ell < 0 or not ell.is_integer():
        raise ValueError("l should be a positive integer")
    if not m.is_integer() or abs(m) > ell:
        raise ValueError("m should be an integer |m| <= l")

    # filename
    func_name = f"spherical_harmonic{filename_args(ell, 'l')}{filename_args(m, 'm')}"

    # create flm
    flm = np.zeros((L * L), dtype=complex)
    ind = ssht.elm2ind(ell, m)
    flm[ind] = 1

    return flm, func_name, config


def squashed_gaussian(args: List[int] = [-2.0, -1.0]) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # args
    try:
        t_sig, freq = args.pop(0), args.pop(0)
    except IndexError:
        raise ValueError("function requires exactly two extra args")

    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L, reality = config["L"], config["reality"]

    # validation
    if not t_sig.is_integer():
        raise ValueError("theta sigma should be an integer")
    if not freq.is_integer():
        raise ValueError("sine frequency should be an integer")
    t_sig, freq = 10 ** t_sig, 10 ** freq

    # filename
    func_name = (
        f"squashed_gaussian{filename_args(t_sig, 'tsig')}"
        f"{filename_args(freq, 'freq')}"
    )

    def grid_fun(
        theta: np.ndarray,
        phi: np.ndarray,
        theta_0: float = 0,
        theta_sig: float = t_sig,
        freq: float = freq,
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(-(((theta - theta_0) / theta_sig) ** 2) / 2) * np.sin(freq * phi)
        return f

    thetas, phis = ssht.sample_positions(L, Grid=True, Method="MWSS")
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, L, Reality=reality, Method="MWSS")

    return flm, func_name, config


# --------------------------------------
# ---------- convolution maps ----------
# --------------------------------------


def earth() -> Tuple[np.ndarray, str, dict]:
    """
    """
    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L = config["L"]

    # filename
    func_name = "earth"

    # extract flm
    matfile = os.path.join(__location__, "data", "EGM2008_Topography_flms_L2190")
    mat_contents = sio.loadmat(matfile)
    flm = np.ascontiguousarray(mat_contents["flm"][:, 0])

    # fill in negative m components so as to
    # avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])

    # don't take the full L
    # invert dataset as Earth backwards
    flm = np.conj(flm[: L * L])

    return flm, func_name, config


def wmap_helper(file_ending: str) -> Tuple[np.ndarray, str, dict]:
    """
    """
    # setup
    config = asdict(Config())
    extra = dict(reality=True)
    config = {**config, **extra}
    L = config["L"]

    # filename
    func_name = "wmap"

    # create flm
    matfile = os.path.join(
        os.environ["SSHT"], "src", "matlab", "data", f"wmap{file_ending}"
    )
    mat_contents = sio.loadmat(matfile)
    cl = np.ascontiguousarray(mat_contents["cl"][:, 0])

    # same random seed
    np.random.seed(0)

    # Simulate CMB in harmonic space.
    flm = np.zeros((L * L), dtype=complex)
    for ell in range(2, L):
        cl[ell - 1] = cl[ell - 1] * 2 * np.pi / (ell * (ell + 1))
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            if m == 0:
                flm[ind] = np.sqrt(cl[ell - 1]) * np.random.randn()
            else:
                flm[ind] = (
                    np.sqrt(cl[ell - 1] / 2) * np.random.randn()
                    + 1j * np.sqrt(cl[ell - 1] / 2) * np.random.randn()
                )

    return flm, func_name, config


def wmap() -> Tuple[np.ndarray, str, dict]:
    """
    """
    # file_ending = '_lcdm_pl_model_yr1_v1'
    # file_ending = '_tt_spectrum_7yr_v4p1'
    file_ending = "_lcdm_pl_model_wmap7baoh0"
    return wmap_helper(file_ending)


# ----------------------------------------------
# ---------- Validate argparse inputs ----------
# ----------------------------------------------


def valid_kernels(func_name: str) -> str:
    """
    check if valid kernel
    """
    if func_name in functions:
        return func_name
    else:
        raise ValueError("Not a valid kernel name to convolve")


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    # check if valid function
    if func_name in functions:
        return func_name
    else:
        raise ValueError("Not a valid function name to plot")


kernels = {
    "dirac_delta": dirac_delta,
    "elongated_gaussian": elongated_gaussian,
    "gaussian": gaussian,
    "harmonic_gaussian": harmonic_gaussian,
    "identity": identity,
    "slepian": slepian,
    "spherical_harmonic": spherical_harmonic,
    "squashed_gaussian": squashed_gaussian,
}
maps = {"earth": earth, "wmap": wmap}
# form dictionary of all functions
functions = {**kernels, **maps}

if __name__ == "__main__":
    # initialise to None
    glm, glm_name = None, None

    args = read_args()
    flm_input = functions[args.flm]
    glm_input = functions.get(args.convolve)
    # if not a convolution
    if glm_input is None:
        num_args = flm_input.__code__.co_argcount
        if args.extra_args is None or num_args == 0:
            flm, flm_name, config = flm_input()
        else:
            flm, flm_name, config = flm_input(args.extra_args)
    # if convolution then flm is a map so no extra args
    else:
        flm, flm_name, _ = flm_input()
        num_args = glm_input.__code__.co_argcount
        if args.extra_args is None or num_args == 0:
            glm, glm_name, config = glm_input()
        else:
            glm, glm_name, config = glm_input(args.extra_args)

    # if using input from argparse
    config["annotation"] = args.annotation
    config["routine"] = args.routine
    config["type"] = args.type

    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)
    sc.plot(args.alpha, args.beta, args.gamma)
