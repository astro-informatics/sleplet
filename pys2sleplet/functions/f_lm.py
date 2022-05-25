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

    def rotate(self, alpha: float, beta: float, *, gamma: float = 0) -> np.ndarray:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def _translation_helper(self, alpha: float, beta: float) -> np.ndarray:
        return ssht.create_ylm(beta, alpha, self.L).conj().flatten()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise is not None:
            self.unnoised_coefficients = self.coefficients.copy()
            nlm = create_noise(self.L, self.coefficients, self.noise)
            self.snr = compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients += nlm

    @abstractmethod
    def _create_coefficients(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
