from typing import Optional

import numpy as np
import pyssht as ssht
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max
from scipy.special import gammaln

from sleplet.functions.f_lm import F_LM
from sleplet.utils.logger import logger
from sleplet.utils.string_methods import (
    convert_camel_case_to_snake_case,
    filename_args,
    wavelet_ending,
)
from sleplet.utils.validation import Validation
from sleplet.utils.wavelet_methods import create_kappas


@dataclass(config=Validation, kw_only=True)
class Ridgelets(F_LM):
    B: int = 3
    j_min: int = 2
    j: Optional[int] = None
    spin: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> np.ndarray:
        logger.info("start computing wavelets")
        self._create_wavelets()
        logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelets[jth]

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{filename_args(self.spin, 'spin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 4
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.j = self.extra_args

    def _create_wavelets(self) -> None:
        """
        compute all wavelets
        """
        ring_lm = self._compute_ring()
        kappas = create_kappas(self.L, self.B, self.j_min)
        self.wavelets = np.zeros((kappas.shape[0], self.L**2), dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            self.wavelets[0, ind] = kappas[0, ell] * ring_lm[ind]
            self.wavelets[1:, ind] = kappas[1:, ell] * ring_lm[ind] / np.sqrt(2 * np.pi)

    def _compute_ring(self) -> np.ndarray:
        """
        compute ring in harmonic space
        """
        ring_lm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(abs(self.spin), self.L):
            logp2 = (
                gammaln(ell + self.spin + 1)
                - ell * np.log(2)
                - gammaln((ell + self.spin) / 2 + 1)
                - gammaln((ell - self.spin) / 2 + 1)
            )
            p0 = np.real((-1) ** ((ell + self.spin) / 2)) * np.exp(logp2)
            ind = ssht.elm2ind(ell, 0)
            ring_lm[ind] = (
                2
                * np.pi
                * np.sqrt((2 * ell + 1) / (4 * np.pi))
                * p0
                * (-1) ** self.spin
                * np.sqrt(
                    np.exp(gammaln(ell - self.spin + 1) - gammaln(ell + self.spin + 1))
                )
            )
        return ring_lm

    @validator("j")
    def check_j(cls, v, values):
        j_max = pys2let_j_max(values["B"], values["L"], values["j_min"])
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}"
            )
        return v
