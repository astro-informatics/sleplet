from typing import Optional

import numpy as np


def sifting_convolution(
    f_coefficient: np.ndarray, g_coefficient: np.ndarray, shannon: Optional[int] = None
) -> np.ndarray:
    """
    computes the sifting convolution between two multipoles
    """
    n = shannon if shannon is not None else np.newaxis
    # change shape if the sizes don't match
    g_reshape = (
        np.reshape(g_coefficient, (-1, 1))
        if len(f_coefficient.shape) != len(g_coefficient.shape)
        else g_coefficient
    )
    return (f_coefficient.T[:n] * g_reshape.conj()[:n]).T
