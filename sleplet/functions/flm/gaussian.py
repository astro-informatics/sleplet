import numpy as np
import pyssht as ssht
from pydantic.dataclasses import dataclass

from sleplet.functions.f_lm import F_LM
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class Gaussian(F_LM):
    sigma: float = 10

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma**2))
        self.coefficients = flm

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.sigma, 'sig')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.sigma = np.float_power(10, self.extra_args[0])
