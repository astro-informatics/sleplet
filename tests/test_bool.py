import numpy as np

import sleplet

J = 2
J_MIN = 0
PHI_0 = np.pi / 6
PHI_1 = np.pi / 3
THETA_0 = np.pi / 6
THETA_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


def test_bool_polar_cap() -> None:
    """Verifies that one is case is a polar cap and one isn't."""
    assert sleplet._bool_methods.is_polar_cap(
        sleplet._vars.PHI_MIN_DEFAULT,
        sleplet._vars.PHI_MAX_DEFAULT,
        sleplet._vars.THETA_MIN_DEFAULT,
        THETA_MAX,
    )
    assert not sleplet._bool_methods.is_polar_cap(PHI_0, PHI_1, THETA_0, THETA_1)


def test_bool_lim_lat_lon() -> None:
    """Verifies that one is case is a polar cap and one isn't."""
    assert sleplet._bool_methods.is_limited_lat_lon(PHI_0, PHI_1, THETA_0, THETA_1)
    assert not sleplet._bool_methods.is_limited_lat_lon(
        sleplet._vars.PHI_MIN_DEFAULT,
        sleplet._vars.PHI_MAX_DEFAULT,
        sleplet._vars.THETA_MIN_DEFAULT,
        THETA_MAX,
    )
