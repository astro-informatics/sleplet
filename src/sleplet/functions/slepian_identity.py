"""Contains the `SlepianIdentity` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.fp import Fp


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianIdentity(Fp):
    """Create an identify function in the Slepian region."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        return np.ones(self.L**2, dtype=np.complex_)

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region._name_ending}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)
