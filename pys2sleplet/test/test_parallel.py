import pytest
from numpy.testing import assert_array_equal

from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, THETA_MAX
from pys2sleplet.utils.string_methods import angle_as_degree


def test_slepian_polar_cap_serial_equal_to_parallel() -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian polar cap give the same result
    """
    serial = SlepianPolarCap(L, THETA_MAX, order=ORDER, ncpu=1)
    parallel = SlepianPolarCap(L, THETA_MAX, order=ORDER)
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)


@pytest.mark.slow
def test_slepian_arbitrary_serial_equal_to_parallel() -> None:
    """
    ensures that the serial and parallel calculation of a given
    Slepian arbitrary region give the same result
    """
    mask_name = f"polar{angle_as_degree(THETA_MAX)}"
    serial = SlepianArbitrary(L, mask_name, ncpu=1)
    parallel = SlepianArbitrary(L, mask_name)
    assert_array_equal(serial.eigenvalues, parallel.eigenvalues)
    assert_array_equal(serial.eigenvectors, parallel.eigenvectors)
