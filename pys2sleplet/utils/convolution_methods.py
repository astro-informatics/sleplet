import numpy as np


def sifting_convolution(flm: np.ndarray, glm: np.ndarray) -> np.ndarray:
    """
    computes the sifting convolution between two multipoles
    """
    return flm * glm.conj()
