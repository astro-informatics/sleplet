"""Contains the `Wmap` class."""

from __future__ import annotations

import typing

import pydantic

import sleplet._data
import sleplet._string_methods
import sleplet._validation
import sleplet.functions.flm

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import typing_extensions


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Wmap(sleplet.functions.flm.Flm):
    """Create the WMAP data."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex128 | np.float64]:
        return sleplet._data.create_wmap_flm.create_flm(self.L)

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
