import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.utils.string_methods import convert_camel_case_to_snake_case


@dataclass(kw_only=True)
class SlepianIdentity(F_P):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        self.coefficients = np.ones(self.L**2, dtype=np.complex_)

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
