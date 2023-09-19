"""Contains the `SlepianWavelets` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import s2wav

import sleplet._string_methods
import sleplet._validation
import sleplet.wavelet_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class SlepianWavelets(Fp):
    """Creates the Slepian wavelets."""

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
            f"_{self.slepian.region.name_ending}"
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

    def _create_wavelets(self) -> npt.NDArray[np.float_]:
        """Computes wavelets in Slepian space."""
        return sleplet.wavelet_methods.create_kappas(self.L**2, self.B, self.j_min)

    @pydantic.field_validator("j")
    def _check_j(cls, v, info: pydantic.FieldValidationInfo):
        j_max = s2wav.utils.shapes.j_max(
            info.data["L"] ** 2,
            info.data["B"],
        )
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - info.data["j_min"]:
            raise ValueError(
                "j should be less than j_max - j_min: "
                f"{j_max - info.data['j_min'] + 1}",
            )
        return v
