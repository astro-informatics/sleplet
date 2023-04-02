import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max

import sleplet
import sleplet._string_methods
import sleplet._validation
import sleplet.functions
import sleplet.slepian
import sleplet.wavelet_methods
from sleplet.functions.fp import FP


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class SlepianWaveletCoefficientsSouthAmerica(FP):
    """TODO."""

    B: int = 3
    """TODO"""
    j_min: int = 2
    """TODO"""
    j: int | None = None
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if (
            isinstance(self.region, sleplet.slepian.Region)
            and self.region.name_ending != "south_america"
        ):
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        sleplet.logger.info("start computing wavelet coefficients")
        self.wavelets, self.wavelet_coefficients = self._create_wavelet_coefficients()
        sleplet.logger.info("finish computing wavelet coefficients")
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
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_ | np.float_]]:
        """Computes wavelet coefficients in Slepian space."""
        sw = sleplet.functions.SlepianWavelets(
            self.L,
            B=self.B,
            j_min=self.j_min,
            region=self.region,
        )
        sa = sleplet.functions.SlepianSouthAmerica(
            self.L,
            region=self.region,
            smoothing=self.smoothing,
        )
        wavelets = sw.wavelets
        wavelet_coefficients = sleplet.wavelet_methods.slepian_wavelet_forward(
            sa.coefficients,
            wavelets,
            self.slepian.N,
        )
        return wavelets, wavelet_coefficients

    @validator("j")
    def _check_j(cls, v, values):
        j_max = pys2let_j_max(values["B"], values["L"] ** 2, values["j_min"])
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}",
            )
        return v
