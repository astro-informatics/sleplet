"""Contains the `DiracDelta` class."""
import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class DiracDelta(Flm):
    """Creates a Dirac delta."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(s2fft.samples.flm_shape(self.L), dtype=np.complex_)
        for ell in range(self.L):
            flm[ell, self.L - 1] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return s2fft.samples.flm_2d_to_1d(flm, self.L)

    def _create_name(self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )
