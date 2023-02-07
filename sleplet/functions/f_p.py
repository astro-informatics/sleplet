from abc import abstractmethod
from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.functions.coefficients import Coefficients
from sleplet.utils.config import settings
from sleplet.utils.mask_methods import create_default_region
from sleplet.utils.noise import compute_snr, create_slepian_noise
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import choose_slepian_method, compute_s_p_omega_prime


@dataclass
class F_P(Coefficients):
    region: Optional[Region] = None

    def __post_init__(self) -> None:
        self.coefficients: np.ndarray
        self.region = (
            self.region
            if isinstance(self.region, Region)
            else create_default_region(settings)
        )
        self.slepian = choose_slepian_method(self.L, self.region)
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, *, gamma: float = 0) -> np.ndarray:
        raise NotImplementedError("Slepian rotation is not defined")

    def _translation_helper(self, alpha: float, beta: float) -> np.ndarray:
        return compute_s_p_omega_prime(self.L, alpha, beta, self.slepian).conj()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise converted to Slepian space
        """
        if self.noise is not None:
            self.unnoised_coefficients = self.coefficients.copy()
            n_p = create_slepian_noise(
                self.L, self.coefficients, self.slepian, self.noise
            )
            self.snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients += n_p

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
