from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SquashedGaussian(Functions):
    L: int
    extra_args: Optional[List[int]] = field(default=None, init=False)
    _t_sigma = field(default=0.01, init=False, repr=False)
    _freq = field(default=0.1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reality = True

    def _grid_fun(
        self, theta: np.ndarray, phi: np.ndarray, theta_0: float = 0
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(-(((theta - theta_0) / self.t_sigma) ** 2) / 2) * np.sin(
            self.freq * phi
        )
        return f

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.freq = [10 ** x for x in self.extra_args]

    def _create_flm(self) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, self.L, self.reality)
        return flm

    def _create_name(self) -> str:
        name = f"squashed_gaussian{filename_args(self.t_sigma, 'tsig')}{filename_args(self.freq, 'freq')}"
        return name

    def _create_annotations(self) -> List[Dict]:
        pass

    @property
    def t_sigma(self) -> float:
        return self._t_sigma

    @t_sigma.setter
    def t_sigma(self, t_sigma: float) -> None:
        self._t_sigma = t_sigma

    @property
    def freq(self) -> float:
        return self._freq

    @freq.setter
    def freq(self, freq: float) -> None:
        self._freq = freq
