from typing import List, Optional

import numpy as np

from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args, verify_args

from ..functions import Functions


class SquashedGaussian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            verify_args(args, 2)
            t_sigma, freq = [10 ** x for x in args]
        else:
            t_sigma, freq = 0.01, 0.1
        self.t_sigma, self.freq = t_sigma, freq

    def _create_flm(self, L: int) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, L, self.reality)
        return flm

    def _create_name(self) -> str:
        name = f"squashed_gaussian{filename_args(self.t_sigma, 'tsig')}{filename_args(self.freq, 'freq')}"
        return name

    @property
    def t_sigma(self) -> float:
        return self.__t_sigma

    @t_sigma.setter
    def t_sigma(self, var: float) -> None:
        self.__t_sigma = var

    @property
    def freq(self) -> float:
        return self.__freq

    @freq.setter
    def freq(self, var: float) -> None:
        self.__freq = var

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
