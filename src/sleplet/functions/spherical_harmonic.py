"""Contains the `SphericalHarmonic` class."""
import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.harmonic_methods
from sleplet.functions.flm import Flm


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class SphericalHarmonic(Flm):
    """Creates spherical harmonic functions."""

    ell: int = 0
    r"""Degree \(\ell \geq 0\)."""
    m: int = 0
    r"""Order \(\leq |\ell|\)"""

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        ind = ssht.elm2ind(self.ell, self.m)
        return sleplet.harmonic_methods._create_spherical_harmonic(self.L, ind)

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.ell, 'l')}"
            f"{sleplet._string_methods.filename_args(self.m, 'm')}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.ell, self.m = self.extra_args

    @validator("ell")
    def _check_ell(cls, v, values):
        if not isinstance(v, int):
            raise TypeError("ell should be an integer")
        if v < 0:
            raise ValueError("ell should be positive")
        if v >= values["L"]:
            raise ValueError("ell should be less than or equal to L")
        return v

    @validator("m")
    def _check_m(cls, v, values):
        if not isinstance(v, int):
            raise TypeError("m should be an integer")
        if abs(v) > values["ell"]:
            raise ValueError("the magnitude of m should be less than ell")
        return v
