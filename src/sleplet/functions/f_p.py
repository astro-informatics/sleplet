from abc import abstractmethod

import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet._mask_methods import create_default_region
from sleplet._validation import Validation
from sleplet.functions.coefficients import Coefficients
from sleplet.noise import _create_slepian_noise, compute_snr
from sleplet.region import Region
from sleplet.slepian_methods import (
    _compute_s_p_omega_prime,
    choose_slepian_method,
)


@dataclass(config=Validation)
class F_P(Coefficients):  # noqa: N801
    def __post_init_post_parse__(self) -> None:
        self.region: Region | None = (
            self.region if isinstance(self.region, Region) else create_default_region()
        )
        self.slepian = choose_slepian_method(self.L, self.region)
        super().__post_init_post_parse__()

    def rotate(
        self, alpha: float, beta: float, *, gamma: float = 0
    ) -> npt.NDArray[np.complex_]:
        raise NotImplementedError("Slepian rotation is not defined")

    def _translation_helper(
        self, alpha: float, beta: float
    ) -> npt.NDArray[np.complex_]:
        return _compute_s_p_omega_prime(self.L, alpha, beta, self.slepian).conj()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """
        adds Gaussian white noise converted to Slepian space
        """
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            n_p = _create_slepian_noise(
                self.L, self.coefficients, self.slepian, self.noise
            )
            snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients = self.coefficients + n_p
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
