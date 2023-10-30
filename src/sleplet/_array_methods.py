import typing

import numpy as np
import numpy.typing as npt


def fill_upper_triangle_of_hermitian_matrix(matrix: npt.NDArray[typing.Any]) -> None:
    """Use Hermitian matrix symmetry can avoid repeated calculations."""
    i_upper = np.triu_indices(len(matrix), k=1)
    matrix[i_upper] = matrix.T[i_upper].conj()
