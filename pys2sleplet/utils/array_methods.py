import numpy as np


def fill_upper_triangle_of_hermitian_matrix(matrix: np.ndarray) -> None:
    """
    using Hermitian matrix symmetry can avoid repeated calculations
    """
    i_upper = np.triu_indices(matrix.shape[0])
    matrix[i_upper] = matrix.T[i_upper].conj()
