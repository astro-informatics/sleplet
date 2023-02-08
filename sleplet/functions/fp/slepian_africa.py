from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.functions.flm.africa import Africa
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_forward
from sleplet.utils.string_methods import convert_camel_case_to_snake_case
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class SlepianAfrica(F_P):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if isinstance(self.region, Region) and self.region.name_ending != "africa":
            raise RuntimeError("Slepian region selected must be 'africa'")

    def _create_coefficients(self) -> npt.NDArray:
        a = Africa(self.L, smoothing=self.smoothing)
        return slepian_forward(self.L, self.slepian, flm=a.coefficients)

    def _create_name(self) -> str:
        return convert_camel_case_to_snake_case(self.__class__.__name__)

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
