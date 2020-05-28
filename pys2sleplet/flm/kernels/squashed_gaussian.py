from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.harmonic_methods import ensure_f_bandlimited
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.string_methods import filename_args
from pys2sleplet.utils.vars import THETA_0


@dataclass
class SquashedGaussian(Functions):
    _t_sigma: float = field(default=0.01, init=False, repr=False)
    _freq: float = field(default=0.1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, self.L, self.reality)
        return flm

    def _create_name(self) -> str:
        name = (
            "squashed_gaussian"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.freq, 'freq')}"
        )
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.freq = [10 ** x for x in self.extra_args]

    def _grid_fun(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(-(((theta - THETA_0) / self.t_sigma) ** 2) / 2) * np.sin(
            self.freq * phi
        )
        return f

    @property
    def freq(self) -> float:
        return self._freq

    @freq.setter
    def freq(self, freq: float) -> None:
        self._freq = freq
        logger.info(f"freq={freq}")

    @property
    def t_sigma(self) -> float:
        return self._t_sigma

    @t_sigma.setter
    def t_sigma(self, t_sigma: float) -> None:
        self._t_sigma = t_sigma
        logger.info(f"t_sigma={t_sigma}")
