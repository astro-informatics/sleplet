import numpy as np
import pytest
from flm.kernels.dirac_delta import DiracDelta
from plotting.create_plot import Plot
from utils.inputs import config
from utils.logging import logger


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
    f_rot = dd.field

    # translation
    dd = DiracDelta(config.L)
    dd.translate(alpha_pi_frac, beta_pi_frac)
    flm_trans = dd.multipole
    f_trans = dd.field

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
    logger.info("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

    if config.AUTO_OPEN:
        filename = f"{dd.name}_L{config.L}_diff_rot_trans_res{dd.resolution}"
        Plot(f_diff.real, dd.resolution, filename).execute()
