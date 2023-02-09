import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.functions.flm.south_america import SouthAmerica
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_forward
from sleplet.utils.string_methods import convert_camel_case_to_snake_case
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class SlepianSouthAmerica(F_P):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if (
            isinstance(self.region, Region)
            and self.region.name_ending != "south_america"
        ):
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> npt.NDArray[np.complex_]:
        sa = SouthAmerica(self.L, smoothing=self.smoothing)
        return slepian_forward(self.L, self.slepian, flm=sa.coefficients)

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
