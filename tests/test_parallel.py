import numpy as np

import sleplet

L_LARGE = 128
L_SMALL = 16
NCPU = 4


def test_split_L_into_chunks() -> None:  # noqa: N802
    """Ensure vector L split into appropriate number of chunks."""
    chunks = sleplet._parallel_methods.split_arr_into_chunks(L_SMALL, NCPU)
    np.testing.assert_equal(len(chunks), NCPU)
    chunk_length = L_SMALL // NCPU
    for chunk in chunks:
        np.testing.assert_allclose(len(chunk), chunk_length, atol=0)
