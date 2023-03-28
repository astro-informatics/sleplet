from numpy.testing import assert_allclose, assert_equal

from sleplet.utils._parallel_methods import split_arr_into_chunks

L_LARGE = 128
L_SMALL = 16
NCPU = 4


def test_split_L_into_chunks() -> None:  # noqa: N802
    """
    ensure vector L split into appropriate number of chunks
    """
    chunks = split_arr_into_chunks(L_SMALL, NCPU)
    assert_equal(len(chunks), NCPU)
    chunk_length = L_SMALL // NCPU
    for chunk in chunks:
        assert_allclose(len(chunk), chunk_length, atol=0)
