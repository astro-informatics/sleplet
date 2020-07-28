from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.harmonic_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args
from pys2sleplet.utils.vars import THETA_0


@dataclass
class SquashedGaussian(Functions):
    t_sigma: float
    freq: float
    _t_sigma: float = field(default=0.01, init=False, repr=False)
    _freq: float = field(default=0.1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.multipole = ensure_f_bandlimited(self._grid_fun, self.L, self.reality)

    def _create_name(self) -> None:
        self.name = (
            "squashed_gaussian"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.freq, 'freq')}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.freq = [10 ** x for x in self.extra_args]

    def _grid_fun(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        function on the grid
        """
        return np.exp(-(((theta - THETA_0) / self.t_sigma) ** 2) / 2) * np.sin(
            self.freq * phi
        )

    @property  # type: ignore
    def freq(self) -> float:
        return self._freq

    @freq.setter
    def freq(self, freq: float) -> None:
        if isinstance(freq, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            freq = SquashedGaussian._freq
        self._freq = freq

    @property  # type: ignore
    def t_sigma(self) -> float:
        return self._t_sigma

    @t_sigma.setter
    def t_sigma(self, t_sigma: float) -> None:
        if isinstance(t_sigma, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            t_sigma = SquashedGaussian._t_sigma
        self._t_sigma = t_sigma
