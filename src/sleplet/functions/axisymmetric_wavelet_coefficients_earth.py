"""Contains the `AxisymmetricWaveletCoefficientsEarth` class."""
import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pydantic

import pys2let

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.earth
import sleplet.wavelet_methods
from sleplet.functions.earth import Earth
from sleplet.functions.flm import Flm

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class AxisymmetricWaveletCoefficientsEarth(Flm):
    """Creates axisymmetric wavelet coefficients of the Earth."""

    B: int = 3
    r"""The wavelet parameter. Represented as \(\lambda\) in the papers."""
    j_min: int = 2
    r"""The minimum wavelet scale. Represented as \(J_{0}\) in the papers."""
    j: int | None = None
    """Option to select a given wavelet. `None` indicates the scaling function,
    whereas `0` would correspond to the selected `j_min`."""
    # TODO: adjust once https://github.com/pydantic/pydantic/issues/5470 fixed
    _earth: Earth = dataclasses.field(default_factory=lambda: Earth(0), repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        _logger.info("start computing wavelet coefficients")
        self.wavelets, self.wavelet_coefficients = self._create_wavelet_coefficients()
        _logger.info("finish computing wavelet coefficients")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelet_coefficients[jth]

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelet_coefficients(
        self,
    ) -> tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
        """Computes wavelet coefficients of the Earth."""
        wavelets = sleplet.wavelet_methods._create_axisymmetric_wavelets(
            self.L,
            self.B,
            self.j_min,
        )
        self._earth = sleplet.functions.earth.Earth(self.L, smoothing=self.smoothing)
        wavelet_coefficients = sleplet.wavelet_methods.axisymmetric_wavelet_forward(
            self.L,
            self._earth.coefficients,
            wavelets,
        )
        return wavelets, wavelet_coefficients

    @pydantic.field_validator("j")
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
