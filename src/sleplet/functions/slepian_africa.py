"""Contains the `SlepianAfrica` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.africa
import sleplet.slepian.region
import sleplet.slepian_methods
from sleplet.functions.fp import Fp


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianAfrica(Fp):
    """
    Create a Slepian region on the topographic map of the Earth of the
    Africa region.
    """

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()
        if (
            isinstance(self.region, sleplet.slepian.region.Region)
            and self.region._name_ending != "africa"
        ):
            msg = "Slepian region selected must be 'africa'"
            raise RuntimeError(msg)

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        a = sleplet.functions.africa.Africa(self.L, smoothing=self.smoothing)
        return sleplet.slepian_methods.slepian_forward(
            self.L,
            self.slepian,
            flm=a.coefficients,
        )

    def _create_name(self: typing_extensions.Self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)
