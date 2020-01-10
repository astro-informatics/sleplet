from typing import List, Optional

import numpy as np

from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args, verify_args

from ..functions import Functions


class ElongatedGaussian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            verify_args(args, 2)
            t_sigma, p_sigma = [10 ** x for x in args]
        else:
            t_sigma, p_sigma = 1e0, 1e-3
        self.t_sigma, self.p_sigma = t_sigma, p_sigma

    def _create_flm(self, L: int) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, L, self.reality)
        return flm

    def _create_name(self) -> str:
        name = f"elongated_gaussian{filename_args(self.t_sigma, 'tsig')}{filename_args(self.p_sigma, 'psig')}"
        return name

    @property
    def t_sigma(self) -> float:
        return self.__t_sigma

    @t_sigma.setter
    def t_sigma(self, var: int) -> None:
        self.__t_sigma = 10 ** var

    @property
    def p_sigma(self) -> float:
        return self.__p_sigma

    @p_sigma.setter
    def p_sigma(self, var: int) -> None:
        self.__p_sigma = 10 ** var

    def _grid_fun(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        theta_0: float = 0,
        phi_0: float = np.pi,
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(
            -(
                ((theta - theta_0) / self.t_sigma) ** 2
                + ((phi - phi_0) / self.p_sigma) ** 2
            )
            / 2
        )
        return f
