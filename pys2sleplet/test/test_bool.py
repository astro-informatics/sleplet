from pys2sleplet.test.constants import (
    J_MIN,
    PHI_0,
    PHI_1,
    THETA_0,
    THETA_1,
    THETA_MAX,
    J,
)
from pys2sleplet.utils.bool_methods import (
    is_ergodic,
    is_limited_lat_lon,
    is_polar_cap,
    is_small_polar_cap,
)
from pys2sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


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


def test_bool_small_polar_cap() -> None:
    """
    verifies that the polar cap is small
    """
    assert is_small_polar_cap(THETA_MAX)
    assert not is_small_polar_cap(THETA_MAX_DEFAULT)


def test_bool_erodicity() -> None:
    """
    verifies that a function follows ergodicity
    """
    assert not is_ergodic(J_MIN)
    assert is_ergodic(J_MIN, J)
