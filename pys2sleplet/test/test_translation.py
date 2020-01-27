import numpy as np
import pytest

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.logging import logger


@pytest.fixture
def alpha_pi_frac() -> float:
    return 0.75


@pytest.fixture
def beta_pi_frac() -> float:
    return 0.125


@pytest.fixture
def L() -> int:
    """
    above 512 get a memory error from travis
    """
    return 512


@pytest.fixture
def show_plots() -> bool:
    return False


def test_dirac_delta_rotate_translate(
    L, alpha_pi_frac, beta_pi_frac, show_plots
) -> None:
    """
    test to ensure that rotation and translation
    give the same result for the Dirac delta
    """
    # rotation
    dd = DiracDelta(L)
    dd.rotate(alpha_pi_frac, beta_pi_frac)
    flm_rot = dd.multipole
    f_rot = dd.field

    # translation
    dd = DiracDelta(L)
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

    if show_plots:
        filename = f"{dd.name}_L{L}_diff_rot_trans_res{dd.resolution}"
        Plot(f_diff.real, dd.resolution, filename).execute()
