import numpy as np

from sleplet.utils._vars import PHI_MAX_DEFAULT, PHI_MIN_DEFAULT, THETA_MIN_DEFAULT
from sleplet.utils.bool_methods import is_ergodic, is_limited_lat_lon, is_polar_cap

J = 2
J_MIN = 0
PHI_0 = np.pi / 6
PHI_1 = np.pi / 3
THETA_0 = np.pi / 6
THETA_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


def test_bool_polar_cap() -> None:
    """
    verifies that one is case is a polar cap and one isn't
    """
    assert is_polar_cap(PHI_MIN_DEFAULT, PHI_MAX_DEFAULT, THETA_MIN_DEFAULT, THETA_MAX)
    assert not is_polar_cap(PHI_0, PHI_1, THETA_0, THETA_1)


def test_bool_lim_lat_lon() -> None:
    """
    verifies that one is case is a polar cap and one isn't
    """
    assert is_limited_lat_lon(PHI_0, PHI_1, THETA_0, THETA_1)
    assert not is_limited_lat_lon(
        PHI_MIN_DEFAULT, PHI_MAX_DEFAULT, THETA_MIN_DEFAULT, THETA_MAX
    )


def test_bool_erodicity() -> None:
    """
    verifies that a function follows ergodicity
    """
    assert not is_ergodic(J_MIN)
    assert is_ergodic(J_MIN, j=J)
