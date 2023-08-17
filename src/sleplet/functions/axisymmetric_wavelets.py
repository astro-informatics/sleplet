"""Contains the `AxisymmetricWavelets` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import pys2let

import sleplet._string_methods
import sleplet._validation
import sleplet.wavelet_methods
from sleplet.functions.flm import Flm

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class AxisymmetricWavelets(Flm):
    """
    Creates scale-discretised axisymmetric wavelets. As
    seen in <https://doi.org/10.1051/0004-6361/201220729>.
    """

    B: int = 3
    r"""The wavelet parameter. Represented as \(\lambda\) in the papers."""
    j_min: int = 2
    r"""The minimum wavelet scale. Represented as \(J_{0}\) in the papers."""
    j: int | None = None
    """Option to select a given wavelet. `None` indicates the scaling function,
    whereas `0` would correspond to the selected `j_min`."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        _logger.info("start computing wavelets")
        self.wavelets = self._create_wavelets()
        _logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelets[jth]

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelets(self) -> npt.NDArray[np.complex_]:
        """Compute all wavelets."""
        return sleplet.wavelet_methods._create_axisymmetric_wavelets(
            self.L,
            self.B,
            self.j_min,
        )

    @pydantic.validator("j")
    def _check_j(cls, v, info: pydantic.FieldValidationInfo):
        j_max = pys2let.pys2let_j_max(
            info.data["B"],
            info.data["L"],
            info.data["j_min"],
        )
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - info.data["j_min"]:
            raise ValueError(
                "j should be less than j_max - j_min: "
                f"{j_max - info.data['j_min'] + 1}",
            )
        return v
