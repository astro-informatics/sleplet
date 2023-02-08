from typing import Optional

import numpy as np
import pyssht as ssht
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max, wavelet_tiling

from sleplet.functions.f_lm import F_LM
from sleplet.utils.logger import logger
from sleplet.utils.string_methods import (
    convert_camel_case_to_snake_case,
    filename_args,
    wavelet_ending,
)
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class DirectionalSpinWavelets(F_LM):
    B: int = 3
    j_min: int = 2
    j: Optional[int] = None
    N: int = 2
    spin: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelets")
        self._create_wavelets()
        logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        self.coefficients = self.wavelet_coefficients[jth]

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{filename_args(self.spin, 'spin')}"
            f"{filename_args(self.N, 'N')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> None:
        self.reality = self.j is None or self.spin == 0

    def _set_spin(self) -> int:
        return self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 5
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.N, self.j = self.extra_args

    def _create_wavelets(self) -> None:
        """
        compute all wavelets
        """
        phi_l, psi_lm = wavelet_tiling(self.B, self.L, self.N, self.j_min, self.spin)
        self.wavelets = np.zeros((psi_lm.shape[1] + 1, self.L**2), dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            self.wavelets[0, ind] = phi_l[ell]
        self.wavelets[1:] = psi_lm.T

    @validator("j")
    def check_j(cls, v, values):
        cls.j_max = pys2let_j_max(values["B"], values["L"], values["j_min"])
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > cls.j_max - values["j_min"]:
            raise ValueError(
                "j should be less than j_max - j_min: "
                f"{cls.j_max - values['j_min'] + 1}"
            )
        return v
