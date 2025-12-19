import numpy as np

import sleplet

L = 128
SMOOTHING = 2


def test_smoothing_changes_flm() -> None:
    """Test the addition of Gaussian noise changes the coefficients."""
    earth = sleplet.functions.Earth(L)
    earth_smoothed = sleplet.functions.Earth(L, smoothing=SMOOTHING)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        earth.coefficients,
        earth_smoothed.coefficients,
    )
