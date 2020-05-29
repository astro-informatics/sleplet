import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, floats, integers
from numpy.testing import assert_array_equal

from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config


@pytest.fixture(scope="module")
def L() -> int:
    """
    needs to be small for speed
    """
    return 8


def valid_theta_max() -> SearchStrategy[float]:
    """
    theta can be in the range [0, 180]
    however we will restrict theta_max to 60 so it is a cap
    """
    return floats(min_value=np.deg2rad(1), max_value=np.deg2rad(60), exclude_min=True)


def valid_orders() -> SearchStrategy[int]:
    """
    the order of the Dm matrix, needs to be less than L
    """
    return integers(min_value=0, max_value=3)


@settings(max_examples=8, derandomize=True, deadline=None)
@given(theta_max=valid_theta_max(), order=valid_orders())
def test_slepian_polar_cap_serial_equal_to_parallel(L, theta_max, order) -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian polar cap give the same result
    """
    serial = SlepianPolarCap(L, theta_max, order=order, ncpu=1)
    parallel = SlepianPolarCap(L, theta_max, order=order, ncpu=config.NCPU)
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


def test_slepian_lat_lon_serial_equal_to_parallel() -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian limited latitude longitude region give the same result
    """
    pass
