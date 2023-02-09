import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.utils.string_methods import convert_camel_case_to_snake_case
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class SlepianIdentity(F_P):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_]:
        return np.ones(self.L**2, dtype=np.complex_)

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
