from numpy.testing import assert_equal

from pys2sleplet.utils.plot_methods import calc_plot_resolution


def test_resolution_values() -> None:
    """
    verifies the correct resolution is chosen for a given bandlimit
    """
    arguments = [1, 10, 100, 1000]
    output = [64, 80, 800, 2000]
    for c, arg in enumerate(arguments):
        assert_equal(calc_plot_resolution(arg), output[c])
