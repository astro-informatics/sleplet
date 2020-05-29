import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_array_equal

from pys2sleplet.slepian.slepian_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap


@pytest.fixture(scope="module")
def L() -> int:
    """
    needs to be small for speed
    """
    return 8


@pytest.fixture(scope="module")
def filename() -> str:
    return "test"


def valid_theta_min() -> SearchStrategy[int]:
    """
    theta can be in the range [0, 180]
    """
    return integers(min_value=1, max_value=30)


def valid_theta_max() -> SearchStrategy[int]:
    """
    theta can be in the range [0, 180]
    """
    return integers(min_value=31, max_value=60)


def valid_phi_min() -> SearchStrategy[int]:
    """
    phi can be in the range [0, 360)
    """
    return integers(min_value=1, max_value=30)


def valid_phi_max() -> SearchStrategy[int]:
    """
    phi can be in the range [0, 360)
    """
    return integers(min_value=31, max_value=60)


def valid_orders() -> SearchStrategy[int]:
    """
    the order of the Dm matrix, needs to be less than L
    """
    return integers(min_value=0, max_value=7)


def valid_mask() -> SearchStrategy[np.ndarray]:
    """
    creates a random mask for arbitrary region
    """
    return arrays(bool, (9, 16))


@settings(max_examples=8, deadline=None)
@given(theta_max=valid_theta_max(), order=valid_orders())
def test_slepian_polar_cap_serial_equal_to_parallel(L, theta_max, order) -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian polar cap give the same result
    """
    serial = SlepianPolarCap(L, np.deg2rad(theta_max), order=order, ncpu=1)
    parallel = SlepianPolarCap(L, np.deg2rad(theta_max), order=order)
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


@settings(max_examples=1, deadline=None)
@given(
    theta_min=valid_theta_min(),
    theta_max=valid_theta_max(),
    phi_min=valid_phi_min(),
    phi_max=valid_phi_max(),
)
def test_slepian_lat_lon_serial_equal_to_parallel(
    L, theta_min, theta_max, phi_min, phi_max
) -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian limited latitude longitude region give the same result
    """
    serial = SlepianLimitLatLong(
        L,
        np.deg2rad(theta_min),
        np.deg2rad(theta_max),
        np.deg2rad(phi_min),
        np.deg2rad(phi_max),
        ncpu=1,
    )
    parallel = SlepianLimitLatLong(
        L,
        np.deg2rad(theta_min),
        np.deg2rad(theta_max),
        np.deg2rad(phi_min),
        np.deg2rad(phi_max),
    )
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


# @settings(max_examples=2, deadline=None)
# @given(mask=valid_mask())
# def test_slepian_arbitrary_serial_equal_to_parallel(L, mask, filename) -> None:
#     """
#     ensures that the serial and parallel calculation of a given
#     Slepian arbitrary region give the same result
#     """
#     serial = SlepianArbitrary(L, mask, filename, ncpu=1)
#     parallel = SlepianArbitrary(L, mask, filename)
#     assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
#     assert_array_equal(serial.eigenvectors, parallel.eigenvectors)
