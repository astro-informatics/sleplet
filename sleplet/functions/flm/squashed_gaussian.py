import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.functions.f_lm import F_LM
from sleplet.utils.harmonic_methods import ensure_f_bandlimited
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args
from sleplet.utils.validation import Validation
from sleplet.utils.vars import THETA_0


@dataclass(config=Validation, kw_only=True)
class SquashedGaussian(F_LM):
    freq: float = 0.1
    t_sigma: float = 1

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray:
        return ensure_f_bandlimited(self._grid_fun, self.L, self.reality, self.spin)

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.freq, 'freq')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.freq = [np.float_power(10, x) for x in self.extra_args]

    def _grid_fun(self, theta: npt.NDArray, phi: npt.NDArray) -> npt.NDArray:
        """
        function on the grid
        """
        return np.exp(-(((theta - THETA_0) / self.t_sigma) ** 2) / 2) * np.sin(
            self.freq * phi
        )
