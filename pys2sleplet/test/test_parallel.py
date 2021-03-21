from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.test.constants import L_LARGE, L_SMALL, NCPU
from pys2sleplet.utils.parallel_methods import split_arr_into_chunks


def test_split_L_into_chunks() -> None:
    """
    ensure vector L split into appropriate number of chunks
    """
    chunks = split_arr_into_chunks(L_SMALL, NCPU)
    assert_equal(len(chunks), NCPU)
    chunk_length = L_SMALL // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=0)


def test_split_L_into_chunks_Lmin_Lmax() -> None:
    """
    ensure vector L split into appropriate number of chunks with L_min
    """
    L_max = L_LARGE + 1  # want to test odd number
    chunks = split_arr_into_chunks(L_max, NCPU, arr_min=L_SMALL)
    assert_equal(len(chunks), NCPU)
    chunk_length = (L_max - L_SMALL) // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=1)
