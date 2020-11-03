import pyssht as ssht
from numpy.testing import assert_equal

from pys2sleplet.test.constants import L_LARGE as RESOLUTION
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import SAMPLING_SCHEME
from pys2sleplet.utils.harmonic_methods import (
    boost_coefficient_resolution,
    invert_flm_boosted,
)


def test_harmonic_coefficients_padded(random_flm) -> None:
    """
    tests that harmonic coefficients are zero padded for plotting
    """
    boost = RESOLUTION ** 2 - L ** 2
    flm_boosted = boost_coefficient_resolution(random_flm, boost)
    assert_equal(len(flm_boosted), RESOLUTION ** 2)


def test_invert_flm_and_boost(random_flm) -> None:
    """
    tests that the flm has been boosted and has right shape
    """
    n_theta, n_phi = ssht.sample_shape(RESOLUTION, Method=SAMPLING_SCHEME)
    f = invert_flm_boosted(random_flm, L, RESOLUTION, method=SAMPLING_SCHEME)
    assert_equal(f.shape, (n_theta, n_phi))
