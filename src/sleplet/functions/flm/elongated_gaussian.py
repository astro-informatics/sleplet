import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.functions.f_lm import F_LM
from sleplet.utils._validation import Validation
from sleplet.utils._vars import PHI_0, THETA_0
from sleplet.utils.harmonic_methods import _ensure_f_bandlimited
from sleplet.utils.string_methods import (
    _convert_camel_case_to_snake_case,
    filename_args,
)


@dataclass(config=Validation, kw_only=True)
class ElongatedGaussian(F_LM):
    p_sigma: float = 0.1
    t_sigma: float = 1

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return _ensure_f_bandlimited(
            self._grid_fun, self.L, reality=self.reality, spin=self.spin
        )

    def _create_name(self) -> str:
        return (
            f"{_convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.t_sigma, 'tsig')}"
            f"{filename_args(self.p_sigma, 'psig')}"
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
            self.t_sigma, self.p_sigma = (
                np.float_power(10, x) for x in self.extra_args
            )

    def _grid_fun(
        self, theta: npt.NDArray[np.float_], phi: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        function on the grid
        """
        return np.exp(
            -(
                ((theta - THETA_0) / self.t_sigma) ** 2
                + ((phi - PHI_0) / self.p_sigma) ** 2
            )
            / 2
        )
