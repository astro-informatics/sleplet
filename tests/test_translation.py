import numpy as np
import pyssht as ssht
from hypothesis import given, seed, settings
from hypothesis.strategies import SearchStrategy, floats
from numpy.testing import assert_allclose, assert_equal, assert_raises

import sleplet

L = 128
THETA_MAX = np.pi / 3


def valid_alphas() -> SearchStrategy[float]:
    """Alpha can be in the range [0, 2*pi)."""
    return floats(min_value=0, max_value=2, exclude_max=True)


def valid_betas() -> SearchStrategy[float]:
    """Beta can be in the range [0, pi]."""
    return floats(min_value=0, max_value=1)


@seed(sleplet._vars.RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(alpha_pi_frac=valid_alphas(), beta_pi_frac=valid_betas())
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
    assert_allclose(np.abs(dd_trans - dd_rot).mean(), 0, atol=0)


def test_slepian_translation_changes_max_polar(slepian_dirac_delta_polar_cap) -> None:
    """Test to ensure the location of the maximum of a field moves when translated."""
    _, beta = sleplet.plot_methods._calc_nearest_grid_point(
        slepian_dirac_delta_polar_cap.L,
        0,
        THETA_MAX / np.pi,
    )
    sdd_trans = slepian_dirac_delta_polar_cap.translate(
        slepian_dirac_delta_polar_cap.alpha,
        beta,
        shannon=slepian_dirac_delta_polar_cap.slepian.N,
    )
    field = sleplet.slepian_methods.slepian_inverse(
        sdd_trans,
        slepian_dirac_delta_polar_cap.L,
        slepian_dirac_delta_polar_cap.slepian,
    )
    new_max = tuple(np.argwhere(field == field.max())[0])
    thetas, _ = ssht.sample_positions(
        slepian_dirac_delta_polar_cap.L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    assert_raises(
        AssertionError,
        assert_equal,
        slepian_dirac_delta_polar_cap.beta,
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
        slepian_dirac_delta_lim_lat_lon.alpha,
        beta,
        shannon=slepian_dirac_delta_lim_lat_lon.slepian.N,
    )
    field = sleplet.slepian_methods.slepian_inverse(
        sdd_trans,
        slepian_dirac_delta_lim_lat_lon.L,
        slepian_dirac_delta_lim_lat_lon.slepian,
    )
    new_max = tuple(np.argwhere(field == field.max())[0])
    thetas, _ = ssht.sample_positions(
        slepian_dirac_delta_lim_lat_lon.L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    assert_raises(
        AssertionError,
        assert_equal,
        slepian_dirac_delta_lim_lat_lon.beta,
        thetas[new_max],
    )
