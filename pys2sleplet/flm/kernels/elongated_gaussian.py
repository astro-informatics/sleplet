from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args


class ElongatedGaussian(Functions):
    def __init__(self, L: int, args: Optional[List[int]] = None) -> None:
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 2
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.p_sigma = [10 ** x for x in args]
        else:
            self.t_sigma, self.p_sigma = 1.0, 0.001

    def _create_flm(self, L: int) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, L, self.reality)
        return flm

    def _create_name(self) -> str:
        name = (
            "elongated_gaussian"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.p_sigma, 'psig')}"
        )
        return name

    def _create_annotations(self) -> List[Dict]:
        pass

    @property
    def t_sigma(self) -> float:
        return self.__t_sigma

    @t_sigma.setter
    def t_sigma(self, var: float) -> None:
        self.__t_sigma = var

    @property
    def p_sigma(self) -> float:
        return self.__p_sigma

    @p_sigma.setter
    def p_sigma(self, var: float) -> None:
        self.__p_sigma = var

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
