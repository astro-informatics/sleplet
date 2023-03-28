from numpy.testing import assert_array_equal, assert_raises

from sleplet.functions.flm.earth import Earth

L = 128
SMOOTHING = 2


def test_smoothing_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L)
    earth_smoothed = Earth(L, smoothing=SMOOTHING)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_smoothed.coefficients,
    )
