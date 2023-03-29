import numpy as np
from numpy.testing import assert_equal

from sleplet._string_methods import (
    _angle_as_degree,
    _filename_angle,
    _multiples_of_pi,
    _wavelet_ending,
    filename_args,
)

J_MIN = 0
PHI_0 = np.pi / 6
PHI_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


def test_add_extra_args_to_filename() -> None:
    """
    test extra args added to filename
    """
    arguments = [0, 123, 0.004]
    output = ["_0a", "_123a", "_1a250"]
    for c, arg in enumerate(arguments):
        assert_equal(filename_args(arg, "a"), output[c])


def test_add_angle_to_filename() -> None:
    """
    tests angle added to filename
    """
    arguments = [(0.75, 0.125, 0), (0.1, 0.2, 0.3), (0.9, 0, 0), (0, 0.8, 0), (0, 0, 0)]
    output = [
        "alpha3pi4_beta1pi8",
        "alpha1pi10_beta1pi5_gamma3pi10",
        "alpha9pi10_beta0",
        "alpha0_beta4pi5",
        "alpha0_beta0",
    ]
    for c, (alpha, beta, gamma) in enumerate(arguments):
        assert_equal(_filename_angle(alpha, beta, gamma), output[c])


def test_print_multiple_of_pi() -> None:
    """
    tests that the pi prefix is added
    """
    arguments = [0, 1, 2, 2.5]
    output = ["0\u03C0", "\u03C0", "2\u03C0", "2\u03C0"]
    for c, arg in enumerate(arguments):
        assert_equal(_multiples_of_pi(arg * np.pi), output[c])


def test_convert_angle_to_degrees() -> None:
    """
    verifies angles is converted to degree
    """
    arguments = [PHI_0, PHI_1, THETA_MAX]
    output = [30, 60, 40]
    for c, arg in enumerate(arguments):
        assert_equal(_angle_as_degree(arg), output[c])


def test_add_to_wavelet_name() -> None:
    """
    test that the correct ending for wavelets is added
    """
    arguments = [None, 0, 1]
    output = ["_scaling", "_0j", "_1j"]
    for c, arg in enumerate(arguments):
        assert_equal(_wavelet_ending(J_MIN, arg), output[c])
