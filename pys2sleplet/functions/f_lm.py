from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.utils.noise import compute_snr, create_noise
from pys2sleplet.utils.smoothing import apply_gaussian_smoothing


@dataclass  # type:ignore
class F_LM(Coefficients):
    def __post_init__(self) -> None:
        self.coefficients: np.ndarray
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def translate(self, alpha: float, beta: float) -> np.ndarray:
        glm = ssht.create_ylm(beta, alpha, self.L).conj().flatten()
        return (
            glm if self.name == "dirac_delta" else self.convolve(self.coefficients, glm)
        )

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise:
            nlm = create_noise(self.L, self.coefficients, self.noise)
            compute_snr(self.L, self.coefficients, nlm)
            self.coefficients += nlm

    def _smooth_signal(self) -> None:
        """
        applies Gaussian smoothing to the signal
        """
        if self.smoothing:
            self.coefficients = apply_gaussian_smoothing(
                self.coefficients, self.L, self.smoothing
            )

    @abstractmethod
    def _create_annotations(self) -> None:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> None:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the function
        """
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> None:
        """
        sets the reality flag to speed up computations
        """
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> None:
        """
        sets the spin value in computations
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        """
        initialises function specific args
        either default value or user input
        """
        raise NotImplementedError
