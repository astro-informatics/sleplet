from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pys2let as s2let

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.functions.flm.south_america import SouthAmerica
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.string_methods import filename_args, wavelet_ending
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_forward


@dataclass
class SlepianWaveletCoefficientsSouthAmerica(F_P):
    B: int
    j_min: int
    j: Optional[int]
    _B: int = field(default=2, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _j_max: int = field(init=False, repr=False)
    _j_min: int = field(default=0, init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)
    _wavelet_coefficients: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region.name_ending != "south_america":
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelet coefficients")
        self._create_wavelet_coefficients()
        logger.info("finish computing wavelet coefficients")
        jth = 0 if self.j is None else self.j + 1
        self.coefficients = self.wavelet_coefficients[jth]

    def _create_name(self) -> None:
        self.name = (
            "slepian_wavelet_coefficients_south_america"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args[:num_args]

    def _create_wavelets(self) -> None:
        """
        computes wavelets in Slepian space
        """
        kappa0, kappa = s2let.axisym_wav_l(self.B, self.L ** 2, self.j_min)
        self.wavelets = np.concatenate((kappa0[np.newaxis], kappa.T))

    def _create_wavelet_coefficients(self) -> None:
        """
        computes wavelet coefficients in Slepian space
        """
        self._create_wavelets()
        sa = SouthAmerica(self.L, region=self.region)
        sa_p = slepian_forward(self.L, sa.coefficients, self.slepian)
        self.wavelet_coefficients = slepian_wavelet_forward(
            sa_p, self.wavelets, self.slepian.N
        )

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = SlepianWaveletCoefficientsSouthAmerica._B
        self._B = B

    @property  # type:ignore
    def j(self) -> Optional[int]:
        return self._j

    @j.setter
    def j(self, j: Optional[int]) -> None:
        if isinstance(j, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j = SlepianWaveletCoefficientsSouthAmerica._j
        self.j_max = s2let.pys2let_j_max(self.B, self.L ** 2, self.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > self.j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {self.j_max - self.j_min + 1}"
            )
        self._j = j

    @property
    def j_max(self) -> int:
        return self._j_max

    @j_max.setter
    def j_max(self, j_max: int) -> None:
        self._j_max = j_max

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = SlepianWaveletCoefficientsSouthAmerica._j_min
        self._j_min = j_min

    @property
    def wavelets(self) -> np.ndarray:
        return self._wavelets

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets

    @property
    def wavelet_coefficients(self) -> np.ndarray:
        return self._wavelet_coefficients

    @wavelet_coefficients.setter
    def wavelet_coefficients(self, wavelet_coefficients: np.ndarray) -> None:
        self._wavelet_coefficients = wavelet_coefficients
