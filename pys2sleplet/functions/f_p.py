from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mask_methods import create_default_region
from pys2sleplet.utils.noise import compute_snr, create_slepian_noise
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import (
    choose_slepian_method,
    compute_s_p_omega_prime,
)


@dataclass  # type:ignore
class F_P(Coefficients):
    region: Optional[Region]
    _region: Optional[Region] = field(default=None, init=False, repr=False)
    _slepian: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.coefficients: np.ndarray  # mypy
        self.region = (
            create_default_region(settings)
            if not isinstance(self.region, Region)
            else self.region
        )
        self.slepian = choose_slepian_method(self.L, self.region)
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        raise NotImplementedError("Slepian rotation is not defined")

    def _translation_helper(self, alpha: float, beta: float) -> np.ndarray:
        return compute_s_p_omega_prime(self.L, alpha, beta, self.slepian).conj()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise converted to Slepian space
        """
        if self.noise is not None:
            n_p = create_slepian_noise(
                self.L, self.coefficients, self.slepian, self.noise
            )
            self.snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients += n_p

    @property  # type:ignore
    def region(self) -> Optional[Region]:
        return self._region

    @region.setter
    def region(self, region: Optional[Region]) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = F_P._region
        self._region = region

    @property
    def slepian(self) -> SlepianFunctions:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian

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
