import hypothesis
import numpy as np
import s2fft

import sleplet

L = 128
THETA_MAX = np.pi / 3


def valid_alphas() -> hypothesis.strategies.SearchStrategy[float]:
    """Alpha can be in the range [0, 2*pi)."""
    return hypothesis.strategies.floats(min_value=0, max_value=2, exclude_max=True)


def valid_betas() -> hypothesis.strategies.SearchStrategy[float]:
    """Beta can be in the range [0, pi]."""
    return hypothesis.strategies.floats(min_value=0, max_value=1)


@hypothesis.seed(sleplet._vars.RANDOM_SEED)
@hypothesis.settings(max_examples=8, deadline=None)
@hypothesis.given(alpha_pi_frac=valid_alphas(), beta_pi_frac=valid_betas())
def test_dirac_delta_rotate_translate(alpha_pi_frac, beta_pi_frac) -> None:
    """
    Test to ensure that rotation and translation
    give the same result for the Dirac delta.
    """
    dd = sleplet.functions.DiracDelta(L)
    alpha, beta = sleplet.plot_methods._calc_nearest_grid_point(
        L,
        alpha_pi_frac,
        beta_pi_frac,
    )
    dd_rot = dd.rotate(alpha, beta)
    dd_trans = dd.translate(alpha, beta)
    np.testing.assert_allclose(np.abs(dd_trans - dd_rot).mean(), 0, atol=0)


def test_slepian_translation_changes_max_polar(slepian_dirac_delta_polar_cap) -> None:
    """Test to ensure the location of the maximum of a field moves when translated."""
    _, beta = sleplet.plot_methods._calc_nearest_grid_point(
        slepian_dirac_delta_polar_cap.L,
        0,
        THETA_MAX / np.pi,
    )
    sdd_trans = slepian_dirac_delta_polar_cap.translate(
        slepian_dirac_delta_polar_cap._alpha,
        beta,
        shannon=slepian_dirac_delta_polar_cap.slepian.N,
    )
    field = sleplet.slepian_methods.slepian_inverse(
        sdd_trans,
        slepian_dirac_delta_polar_cap.L,
        slepian_dirac_delta_polar_cap.slepian,
    )
    new_max = tuple(np.argwhere(field == field.max())[0])
    thetas = np.tile(
        s2fft.samples.thetas(
            slepian_dirac_delta_polar_cap.L,
            sampling=sleplet._vars.SAMPLING_SCHEME,
        ),
        (s2fft.samples.nphi_equiang(L, sampling=sleplet._vars.SAMPLING_SCHEME), 1),
    ).T
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_equal,
        slepian_dirac_delta_polar_cap._beta,
        thetas[new_max],
    )


def test_slepian_translation_changes_max_lim_lat_lon(
    slepian_dirac_delta_lim_lat_lon,
) -> None:
    """Test to ensure the location of the maximum of a field moves when translated."""
    _, beta = sleplet.plot_methods._calc_nearest_grid_point(
        slepian_dirac_delta_lim_lat_lon.L,
        0,
        THETA_MAX / np.pi,
    )
    sdd_trans = slepian_dirac_delta_lim_lat_lon.translate(
        slepian_dirac_delta_lim_lat_lon._alpha,
        beta,
        shannon=slepian_dirac_delta_lim_lat_lon.slepian.N,
    )
    field = sleplet.slepian_methods.slepian_inverse(
        sdd_trans,
        slepian_dirac_delta_lim_lat_lon.L,
        slepian_dirac_delta_lim_lat_lon.slepian,
    )
    new_max = tuple(np.argwhere(field == field.max())[0])
    thetas = np.tile(
        s2fft.samples.thetas(
            slepian_dirac_delta_lim_lat_lon.L,
            sampling=sleplet._vars.SAMPLING_SCHEME,
        ),
        (s2fft.samples.nphi_equiang(L, sampling=sleplet._vars.SAMPLING_SCHEME), 1),
    ).T
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_equal,
        slepian_dirac_delta_lim_lat_lon._beta,
        thetas[new_max],
    )
