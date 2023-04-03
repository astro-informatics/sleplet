import pyssht as ssht
from numpy.testing import assert_equal

import sleplet

L_LARGE = 128
L_SMALL = 16


def test_harmonic_coefficients_padded(random_flm) -> None:
    """Tests that harmonic coefficients are zero padded for plotting."""
    boost = L_LARGE**2 - L_SMALL**2
    flm_boosted = sleplet.harmonic_methods._boost_coefficient_resolution(
        random_flm,
        boost,
    )
    assert_equal(len(flm_boosted), L_LARGE**2)


def test_invert_flm_and_boost(random_flm) -> None:
    """Tests that the flm has been boosted and has right shape."""
    n_theta, n_phi = ssht.sample_shape(L_LARGE, Method=sleplet._vars.SAMPLING_SCHEME)
    f = sleplet.harmonic_methods.invert_flm_boosted(random_flm, L_SMALL, L_LARGE)
    assert_equal(f.shape, (n_theta, n_phi))
