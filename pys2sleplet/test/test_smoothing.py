from numpy.testing import assert_array_equal, assert_raises

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.test.constants import L_LARGE


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L_LARGE)
    earth_smoothed = Earth(L_LARGE, smoothed=True)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_smoothed.coefficients,
    )
