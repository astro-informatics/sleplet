from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class HarmonicGaussian(Functions):
    _l_sigma: float = field(default=1_000, init=False, repr=False)
    _m_sigma: float = field(default=1_000, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                self.flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)

    def _create_name(self) -> None:
        self.Lname = (
            "harmonic_gaussian"
            f"{filename_args(self.l_sigma, 'lsig')}"
            f"{filename_args(self.m_sigma, 'msig')}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.l_sigma, self.m_sigma = [10 ** x for x in self.extra_args]

    @property
    def l_sigma(self) -> float:
        return self._l_sigma

    @l_sigma.setter
    def l_sigma(self, l_sigma: float) -> None:
        self._l_sigma = l_sigma
        logger.info(f"l_sigma={l_sigma}")

    @property
    def m_sigma(self) -> float:
        return self._m_sigma

    @m_sigma.setter
    def m_sigma(self, m_sigma: float) -> None:
        self._m_sigma = m_sigma
        logger.info(f"m_sigma={m_sigma}")
