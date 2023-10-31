"""Contains the `Gaussian` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class Gaussian(Flm):
    r"""Create a Gaussian \(\exp(-\frac{{\ell}^{2}}{2\sigma^{2}})\)."""

    sigma: float = 10
    r"""Sets the \(\sigma\) value."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma**2))
        return flm

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.sigma, 'sig')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return True

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.sigma = np.float_power(10, self.extra_args[0])
