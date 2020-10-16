from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import (
    choose_slepian_method,
    compute_s_p_omega_prime,
    slepian_forward,
    slepian_inverse,
)

_file_location = Path(__file__).resolve()


@dataclass  # type:ignore
class F_P(Coefficients):
    rank: int
    region: Optional[Region]
    _rank: int = field(default=0, init=False, repr=False)
    _region: Optional[Region] = field(default=None, init=False, repr=False)
    _slepian: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = (
            Region(
                gap=settings.POLAR_GAP,
                mask_name=settings.SLEPIAN_MASK,
                phi_max=np.deg2rad(settings.PHI_MAX),
                phi_min=np.deg2rad(settings.PHI_MIN),
                theta_max=np.deg2rad(settings.THETA_MAX),
                theta_min=np.deg2rad(settings.THETA_MIN),
            )
            if self.region is None
            else self.region
        )
        self.slepian = choose_slepian_method(self.L, self.region)
        super().__post_init__()

    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        return slepian_inverse(self.L, coefficients, self.slepian)

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        f = self.inverse(self.coefficients)
        flm = ssht.forward(f, self.L)
        flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, self.L)
        return slepian_forward(self.L, flm_rot, self.slepian)

    def translate(self, alpha: float, beta: float) -> np.ndarray:
        gp = compute_s_p_omega_prime(self.L, alpha, beta, self.slepian).conj()
        return self.convolve(self.coefficients, gp)

    def _add_noise_to_signal(self) -> None:
        pass

    def _smooth_signal(self) -> None:
        pass

    @property  # type:ignore
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        if isinstance(rank, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            rank = F_P._rank
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self._rank = rank

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
    def _create_annotations(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
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
