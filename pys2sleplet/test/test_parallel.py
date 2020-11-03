import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_LARGE
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import NCPU, ORDER, THETA_MAX
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
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


def test_split_L_into_chunks() -> None:
    """
    ensure vector L split into appropriate number of chunks
    """
    chunks = split_L_into_chunks(L, NCPU)
    assert len(chunks) == NCPU
    chunk_length = L // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=1)


def test_split_L_into_chunks_Lmin_Lmax() -> None:
    """
    ensure vector L split into appropriate number of chunks with L_min
    """
    L_max = L_LARGE + 1  # want to test odd number
    chunks = split_L_into_chunks(L_max, NCPU, L_min=L)
    assert len(chunks) == NCPU
    chunk_length = (L_max - L) // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=1)
