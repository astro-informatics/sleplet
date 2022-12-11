from numpy.testing import assert_array_equal, assert_raises

from sleplet.functions.flm.earth import Earth
from sleplet.test.constants import L_LARGE
from sleplet.utils.vars import SMOOTHING


def test_smoothing_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L_LARGE)
    earth_smoothed = Earth(L_LARGE, smoothing=SMOOTHING)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_smoothed.coefficients,
    )
