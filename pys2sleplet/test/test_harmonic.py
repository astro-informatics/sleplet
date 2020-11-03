import numpy as np
from numpy.testing import assert_equal

from pys2sleplet.test.constants import L_LARGE as RESOLUTION
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.harmonic_methods import boost_coefficient_resolution


def test_harmonic_coefficients_padded() -> None:
    """
    tests that harmonic coefficients are zero padded for plotting
    """
    flm = np.ones(L ** 2)
    boost = RESOLUTION ** 2 - L ** 2
    flm_boosted = boost_coefficient_resolution(flm, boost)
    assert_equal(len(flm_boosted), RESOLUTION ** 2)
