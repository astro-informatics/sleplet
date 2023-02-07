from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from sleplet.functions.f_lm import F_LM
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args


@dataclass
class HarmonicGaussian(F_LM):
    l_sigma: float
    m_sigma: float
    _l_sigma: float = field(default=10, init=False, repr=False)
    _m_sigma: float = field(default=10, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)
        self.coefficients = flm

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.l_sigma, 'lsig')}"
            f"{filename_args(self.m_sigma, 'msig')}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.l_sigma, self.m_sigma = [
                np.float_power(10, x) for x in self.extra_args
            ]
