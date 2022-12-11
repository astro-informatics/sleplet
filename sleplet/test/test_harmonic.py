import pyssht as ssht
from numpy.testing import assert_equal

from sleplet.test.constants import L_LARGE, L_SMALL
from sleplet.utils.harmonic_methods import (
    boost_coefficient_resolution,
    invert_flm_boosted,
)
from sleplet.utils.vars import SAMPLING_SCHEME


def test_harmonic_coefficients_padded(random_flm) -> None:
    """
    tests that harmonic coefficients are zero padded for plotting
    """
    boost = L_LARGE**2 - L_SMALL**2
    flm_boosted = boost_coefficient_resolution(random_flm, boost)
    assert_equal(len(flm_boosted), L_LARGE**2)


def test_invert_flm_and_boost(random_flm) -> None:
    """
    tests that the flm has been boosted and has right shape
    """
    n_theta, n_phi = ssht.sample_shape(L_LARGE, Method=SAMPLING_SCHEME)
    f = invert_flm_boosted(random_flm, L_SMALL, L_LARGE)
    assert_equal(f.shape, (n_theta, n_phi))
