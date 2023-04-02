"""Contains the `SlepianSouthAmerica` class."""
import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.south_america
import sleplet.slepian.region
from sleplet.functions.fp import Fp


@dataclass(config=sleplet._validation.Validation)
class SlepianSouthAmerica(Fp):
    """
    Creates a Slepian region on the topographic map of the Earth of the
    South America region.
    """

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if (
            isinstance(self.region, sleplet.slepian.region.Region)
            and self.region.name_ending != "south_america"
        ):
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        sa = sleplet.functions.south_america.SouthAmerica(
            self.L,
            smoothing=self.smoothing,
        )
        return sleplet.slepian_methods.slepian_forward(
            self.L,
            self.slepian,
            flm=sa.coefficients,
        )

    def _create_name(self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )
