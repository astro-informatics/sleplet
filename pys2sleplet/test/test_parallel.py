import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy, floats, integers
from numpy.testing import assert_array_equal

from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, PHI_0, PHI_1, THETA_0, THETA_1


def valid_theta_min() -> SearchStrategy[float]:
    """
    theta can be in the range [0, 180]
    """
    return floats(min_value=0, max_value=THETA_0, exclude_min=True)


def valid_theta_max() -> SearchStrategy[float]:
    """
    theta can be in the range [0, 180]
    """
    return floats(min_value=THETA_0, max_value=THETA_1)


def valid_phi_min() -> SearchStrategy[float]:
    """
    phi can be in the range [0, 360)
    """
    return floats(min_value=0, max_value=PHI_0, exclude_min=True)


def valid_phi_max() -> SearchStrategy[float]:
    """
    phi can be in the range [0, 360)
    """
    return floats(min_value=PHI_0, max_value=PHI_1)


def valid_orders() -> SearchStrategy[int]:
    """
    the order of the Dm matrix, needs to be less than L
    """
    return integers(min_value=0, max_value=L - 1)


def valid_mask() -> SearchStrategy[np.ndarray]:
    """
    creates a random mask for arbitrary region
    """
    return arrays(bool, (L + 1, 2 * L))


@settings(max_examples=8, deadline=None)
@given(theta_max=valid_theta_max(), order=valid_orders())
def test_slepian_polar_cap_serial_equal_to_parallel(theta_max, order) -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian polar cap give the same result
    """
    serial = SlepianPolarCap(L, theta_max, order=ORDER, ncpu=1)
    parallel = SlepianPolarCap(L, theta_max, order=ORDER)
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
    theta_min, theta_max, phi_min, phi_max
) -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian limited latitude longitude region give the same result
    """
    serial = SlepianLimitLatLon(
        L,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
        ncpu=1,
    )
    parallel = SlepianLimitLatLon(
        L, theta_min=theta_min, theta_max=theta_max, phi_min=phi_min, phi_max=phi_max
    )
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


# @settings(max_examples=2, deadline=None)
# @given(mask=valid_mask())
# def test_slepian_arbitrary_serial_equal_to_parallel(mask) -> None:
#     """
#     ensures that the serial and parallel calculation of a given
#     Slepian arbitrary region give the same result
#     """
#     serial = SlepianArbitrary(L, mask, MASK_NAME, ncpu=1)
#     parallel = SlepianArbitrary(L, mask, MASK_NAME)
#     assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
#     assert_array_equal(serial.eigenvectors, parallel.eigenvectors)
