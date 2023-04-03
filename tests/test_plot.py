import cmocean
import numpy as np
from numpy.testing import assert_equal

import sleplet

L = 128
PHI_0 = np.pi / 6
PL_ENTRIES = 255
THETA_MAX = np.pi / 3


def test_resolution_values() -> None:
    """Verifies the correct resolution is chosen for a given bandlimit."""
    arguments = [1, 10, 100, 1000]
    output = [64, 80, 800, 2000]
    for c, arg in enumerate(arguments):
        assert_equal(sleplet.plot_methods.calc_plot_resolution(arg), output[c])


def test_create_colourscale() -> None:
    """Test creates a plotly compatible colourscale."""
    colourscale = sleplet.plot_methods._convert_colourscale(
        cmocean.cm.ice,
        pl_entries=PL_ENTRIES,
    )
    assert_equal(len(colourscale), PL_ENTRIES)


def test_find_nearest_grid_point() -> None:
    """Test to find nearest grid point to provided angles."""
    alpha, beta = sleplet.plot_methods._calc_nearest_grid_point(
        L,
        PHI_0 / np.pi,
        THETA_MAX / np.pi,
    )
    assert_equal(alpha, 0.5154175447295755)
    assert_equal(beta, 1.055378782065321)
