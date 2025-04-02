import numpy as np

import sleplet

L = 128
PHI_0 = np.pi / 6
PL_ENTRIES = 255
THETA_MAX = np.pi / 3


def test_resolution_values() -> None:
    """Verify the correct resolution is chosen for a given bandlimit."""
    arguments = [1, 10, 100, 1000]
    output = [64, 80, 800, 2000]
    for c, arg in enumerate(arguments):
        np.testing.assert_equal(
            sleplet.plot_methods.calc_plot_resolution(arg),
            output[c],
        )


def test_find_nearest_grid_point() -> None:
    """Test to find nearest grid point to provided angles."""
    alpha, beta = sleplet.plot_methods._calc_nearest_grid_point(
        L,
        PHI_0 / np.pi,
        THETA_MAX / np.pi,
    )
    np.testing.assert_equal(alpha, 0.5154175447295755)
    np.testing.assert_equal(beta, 1.055378782065321)
