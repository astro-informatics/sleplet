from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.harmonic_methods import ensure_f_bandlimited
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.string_methods import filename_args
from pys2sleplet.utils.vars import PHI_0, THETA_0


@dataclass
class ElongatedGaussian(Functions):
    _p_sigma: float = field(default=0.001, init=False, repr=False)
    _t_sigma: float = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.flm = ensure_f_bandlimited(self._grid_fun, self.L, self.reality)

    def _create_name(self) -> None:
        self.name = (
            "elongated_gaussian"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.p_sigma, 'psig')}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.p_sigma = [10 ** x for x in self.extra_args]

    def _grid_fun(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(
            -(
                ((theta - THETA_0) / self.t_sigma) ** 2
                + ((phi - PHI_0) / self.p_sigma) ** 2
            )
            / 2
        )
        return f

    @property
    def p_sigma(self) -> float:
        return self._p_sigma

    @p_sigma.setter
    def p_sigma(self, p_sigma: float) -> None:
        self._p_sigma = p_sigma
        logger.info(f"p_sigma={p_sigma}")

    @property
    def t_sigma(self) -> float:
        return self._t_sigma

    @t_sigma.setter
    def t_sigma(self, t_sigma: float) -> None:
        self._t_sigma = t_sigma
        logger.info(f"t_sigma={t_sigma}")
