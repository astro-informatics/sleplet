from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max

from sleplet.functions.f_p import F_P
from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.utils.logger import logger
from sleplet.utils.string_methods import (
    convert_camel_case_to_snake_case,
    filename_args,
    wavelet_ending,
)
from sleplet.utils.wavelet_methods import slepian_wavelet_forward


@dataclass
class SlepianWaveletCoefficientsSouthAmerica(F_P):
    B: int
    j_min: int
    j: Optional[int]
    _B: int = field(default=3, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _j_max: int = field(init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)
    _wavelet_coefficients: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region.name_ending != "south_america":
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelet coefficients")
        self._create_wavelet_coefficients()
        logger.info("finish computing wavelet coefficients")
        jth = 0 if self.j is None else self.j + 1
        self.coefficients = self.wavelet_coefficients[jth]

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
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
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelet_coefficients(self) -> None:
        """
        computes wavelet coefficients in Slepian space
        """
        sw = SlepianWavelets(self.L, B=self.B, j_min=self.j_min, region=self.region)
        sa = SlepianSouthAmerica(self.L, region=self.region, smoothing=self.smoothing)
        self.wavelets = sw.wavelets
        self.wavelet_coefficients = slepian_wavelet_forward(
            sa.coefficients, self.wavelets, self.slepian.N
        )

    @j.setter
    def j(self, j: Optional[int]) -> None:
        self.j_max = pys2let_j_max(self.B, self.L**2, self.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > self.j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {self.j_max - self.j_min + 1}"
            )
        self._j = j
