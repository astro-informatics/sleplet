import numpy as np
from hypothesis import given, seed, settings
from hypothesis.strategies import SearchStrategy, floats
from numpy.testing import assert_allclose

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.test.constants import L_LARGE as L
from pys2sleplet.utils.vars import RANDOM_SEED


def valid_alphas() -> SearchStrategy[float]:
    """
    alpha can be in the range [0, 2*pi)
    """
    return floats(min_value=0, max_value=2, exclude_max=True)


def valid_betas() -> SearchStrategy[float]:
    """
    beta can be in the range [0, pi]
    """
    return floats(min_value=0, max_value=1)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(alpha_pi_frac=valid_alphas(), beta_pi_frac=valid_betas())
def test_dirac_delta_rotate_translate(alpha_pi_frac, beta_pi_frac) -> None:
    """
    test to ensure that rotation and translation
    give the same result for the Dirac delta
    """
    dd = DiracDelta(L)
    dd_rot = dd.rotate(alpha_pi_frac, beta_pi_frac)
    dd_trans = dd.translate(alpha_pi_frac, beta_pi_frac)
    assert_allclose(np.abs(dd_trans - dd_rot).mean(), 0, atol=1e-16)
