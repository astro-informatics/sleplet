import numpy as np
import pytest

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger


@pytest.fixture
def alpha_pi_frac() -> float:
    return 0.75


@pytest.fixture
def beta_pi_frac() -> float:
    return 0.125


def test_dirac_delta_rotate_translate(alpha_pi_frac, beta_pi_frac) -> None:
    """
    test to ensure that rotation and translation
    give the same result for the Dirac delta
    """
    # rotation
    dd = DiracDelta(config.L)
    dd.rotate(alpha_pi_frac, beta_pi_frac)
    flm_rot = dd.multipole
    f_rot, f_rot_plot = dd.field, dd.plot

    # translation
    dd = DiracDelta(config.L)
    dd.translate(alpha_pi_frac, beta_pi_frac)
    flm_trans = dd.multipole
    f_trans, f_trans_plot = dd.field, dd.plot

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot_plot - f_trans_plot

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, rtol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-12)
    logger.info(f"Translation/rotation difference max error: {np.abs(flm_diff).max()}")

    if config.AUTO_OPEN:
        filename = f"{dd.name}_L{config.L}_diff_rot_trans_res{dd.resolution}"
        Plot(f_diff.real, dd.resolution, filename).execute()
