import numpy as np
from numpy import typing as npt


def sifting_convolution(
    f_coefficient: npt.NDArray,
    g_coefficient: npt.NDArray,
    *,
    shannon: int | None = None,
) -> npt.NDArray:
    """
    computes the sifting convolution between two multipoles
    """
    n = shannon if shannon is not None else np.newaxis
    # change shape if the sizes don't match
    g_reshape = (
        g_coefficient[:, np.newaxis]
        if len(g_coefficient.shape) < len(f_coefficient.shape)
        else g_coefficient
    )
    return (f_coefficient.T[:n] * g_reshape.conj()[:n]).T
