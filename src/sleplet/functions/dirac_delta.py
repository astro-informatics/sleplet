"""Contains the `DiracDelta` class."""

from __future__ import annotations

import typing

import numpy as np
import pydantic

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.flm

if typing.TYPE_CHECKING:
    import numpy.typing as npt
    import typing_extensions


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class DiracDelta(sleplet.functions.flm.Flm):
    """Create a Dirac delta."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex128 | np.float64]:
        flm = np.zeros(self.L**2, dtype=np.complex128)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return flm

    def _create_name(self: typing_extensions.Self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return True

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)
