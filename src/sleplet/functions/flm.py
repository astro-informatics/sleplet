"""Contains the abstract `Flm` class."""
from abc import abstractmethod

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._validation
import sleplet.noise
from sleplet.functions.coefficients import Coefficients


@dataclass(config=sleplet._validation.Validation)
class Flm(Coefficients):
    """Abstract parent class to handle harmonic coefficients on the sphere."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def rotate(  # noqa: D102
        self,
        alpha: float,
        beta: float,
        *,
        gamma: float = 0,
    ) -> npt.NDArray[np.complex_]:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def _translation_helper(
        self,
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.complex_]:
        return ssht.create_ylm(beta, alpha, self.L).conj().flatten()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise to the signal."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = sleplet.noise._create_noise(self.L, self.coefficients, self.noise)
            snr = sleplet.noise.compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
            return unnoised_coefficients, snr
        return None, None

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
