from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.utils.noise import compute_snr, create_noise


@dataclass  # type:ignore
class F_LM(Coefficients):
    def __post_init__(self) -> None:
        self.coefficients: np.ndarray  # mypy
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def _translation_helper(self, alpha: float, beta: float) -> np.ndarray:
        return ssht.create_ylm(beta, alpha, self.L).conj().flatten()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise is not None:
            nlm = create_noise(self.L, self.coefficients, self.noise)
            self.snr = compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients += nlm

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
