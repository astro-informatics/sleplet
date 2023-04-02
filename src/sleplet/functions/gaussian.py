"""Contains the `Gaussian` class."""
import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class Gaussian(Flm):
    r"""Creates a Gaussian \(\exp(-\frac{{\ell}^{2}}{2\sigma^{2}})\)."""

    sigma: float = 10
    r"""Sets the \(\sigma\) value."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma**2))
        return flm

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.sigma, 'sig')}"
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
