import numpy as np

import sleplet

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
SNR_IN = 10


def test_adding_noise_changes_flm() -> None:
    """Test the addition of Gaussian noise changes the coefficients."""
    earth = sleplet.functions.Earth(L)
    earth_noised = sleplet.functions.Earth(L, noise=SNR_IN)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        earth.coefficients,
        earth_noised.coefficients,
    )
