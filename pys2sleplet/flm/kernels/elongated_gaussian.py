from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class ElongatedGaussian(Functions):
    L: int
    extra_args: List[int]
    _p_sigma: float = field(init=False, repr=False)
    _t_sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.p_sigma = 0.001
        self.reality = True
        self.t_sigma = 1

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

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 2
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.p_sigma = [10 ** x for x in args]

    def _create_flm(self) -> np.ndarray:
        flm = ensure_f_bandlimited(self._grid_fun, self.L, self.reality)
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
        return self._t_sigma

    @t_sigma.setter
    def t_sigma(self, t_sigma: float) -> None:
        self._t_sigma = t_sigma

    @property
    def p_sigma(self) -> float:
        return self._p_sigma

    @p_sigma.setter
    def p_sigma(self, p_sigma: float) -> None:
        self._p_sigma = p_sigma
