from numpy.testing import assert_array_equal, assert_raises

from sleplet.functions.flm.earth import Earth

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
SNR_IN = 10


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L)
    earth_noised = Earth(L, noise=SNR_IN)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_noised.coefficients,
    )
