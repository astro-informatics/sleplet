from numpy.testing import assert_array_equal, assert_raises

import sleplet

L = 128
SMOOTHING = 2


def test_smoothing_changes_flm() -> None:
    """Tests the addition of Gaussian noise changes the coefficients."""
    earth = sleplet.functions.Earth(L)
    earth_smoothed = sleplet.functions.Earth(L, smoothing=SMOOTHING)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_smoothed.coefficients,
    )
