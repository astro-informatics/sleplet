import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pys2sleplet.flm.kernels.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.flm.kernels.identity import Identity
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger


def test_earth_identity_convolution() -> None:
    """
    test to ensure that the convolving with the
    identity function doesn't change the map
    """
    f = Earth(config.L)
    g = Identity(config.L)
    flm = f.multipole

    f.convolve(g.multipole)
    flm_conv = f.multipole

    assert_array_equal(flm, flm_conv)
    logger.info("Identity convolution passed test")


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map
    """
    f = Earth(config.L)
    g = HarmonicGaussian(config.L)
    flm = f.multipole
    f_map, f_map_plot = f.field, f.field_padded

    f.convolve(g.multipole)
    flm_conv = f.multipole
    f_conv, f_conv_plot = f.field, f.field_padded

    flm_diff = flm - flm_conv
    f_diff = f_map_plot - f_conv_plot

    assert_array_equal(flm, flm_conv)
    assert_allclose(f_map, f_conv, rtol=0.2)
    logger.info(
        "Earth/harmonic gaussian convolution difference max error: "
        f"{np.abs(flm_diff).max()}"
    )

    if config.AUTO_OPEN:
        filename = f"{g.name}_L{f.L}_diff_{f.name}_res{f.resolution}_real"
        Plot(f_diff.real, f.resolution, filename).execute()
