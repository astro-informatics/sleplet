import numpy as np

import s2fft

import sleplet

L_LARGE = 128
L_SMALL = 16


def test_harmonic_coefficients_padded(random_flm) -> None:
    """Tests that harmonic coefficients are zero padded for plotting."""
    boost = L_LARGE**2 - L_SMALL**2
    flm_boosted = s2fft.samples.flm_1d_to_2d(
        sleplet.harmonic_methods._boost_coefficient_resolution(
            s2fft.samples.flm_2d_to_1d(random_flm, L_SMALL),
            boost,
        ),
        L_LARGE,
    )
    np.testing.assert_equal(flm_boosted.shape, s2fft.samples.flm_shape(L_LARGE))


def test_invert_flm_and_boost(random_flm) -> None:
    """Tests that the flm has been boosted and has right shape."""
    n_theta, n_phi = s2fft.samples.f_shape(
        L_LARGE,
        sampling=sleplet._vars.SAMPLING_SCHEME,
    )
    f = sleplet.harmonic_methods.invert_flm_boosted(random_flm, L_SMALL, L_LARGE)
    np.testing.assert_equal(f.shape, (n_theta, n_phi))
