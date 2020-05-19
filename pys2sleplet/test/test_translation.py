import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, floats

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger


def alpha_pi_frac() -> SearchStrategy[float]:
    """
    alpha can be in the range [0, 2*pi)
    """
    return floats(min_value=0, max_value=2, exclude_max=True, width=16)


def beta_pi_frac() -> SearchStrategy[float]:
    """
    beta can be in the range [0, pi]
    """
    return floats(min_value=0, max_value=1, width=16)


@settings(max_examples=8, derandomize=True, deadline=None)
@given(alpha=alpha_pi_frac(), beta=beta_pi_frac())
def test_dirac_delta_rotate_translate(alpha, beta) -> None:
    """
    test to ensure that rotation and translation
    give the same result for the Dirac delta
    """
    # rotation
    dd = DiracDelta(config.L)
    dd.rotate(alpha, beta)
    flm_rot = dd.multipole
    f_rot, f_rot_plot = dd.field, dd.plot

    # translation
    dd = DiracDelta(config.L)
    dd.translate(alpha, beta)
    flm_trans = dd.multipole
    f_trans, f_trans_plot = dd.field, dd.plot

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot_plot - f_trans_plot

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, rtol=1e-13)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-11)
    logger.info(f"Translation/rotation difference max error: {np.abs(flm_diff).max()}")

    if config.AUTO_OPEN:
        filename = f"{dd.name}_L{config.L}_diff_rot_trans_res{dd.resolution}"
        Plot(f_diff.real, dd.resolution, filename).execute()
