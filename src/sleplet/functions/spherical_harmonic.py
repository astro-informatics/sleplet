"""Contains the `SphericalHarmonic` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
import sleplet.harmonic_methods
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class SphericalHarmonic(Flm):
    """Create spherical harmonic functions."""

    ell: int = 0
    r"""Degree \(\ell \geq 0\)."""
    m: int = 0
    r"""Order \(\leq |\ell|\)"""

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        ind = ssht.elm2ind(self.ell, self.m)
        return sleplet.harmonic_methods._create_spherical_harmonic(self.L, ind)

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.ell, 'l')}"
            f"{sleplet._string_methods.filename_args(self.m, 'm')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.ell, self.m = self.extra_args

    @pydantic.field_validator("ell")
    def _check_ell(
        cls,  # noqa: ANN101
        v: int,
        info: pydantic.ValidationInfo,
    ) -> int:
        if not isinstance(v, int):
            msg = "ell should be an integer"
            raise TypeError(msg)
        if v < 0:
            msg = "ell should be positive"
            raise ValueError(msg)
        if v >= info.data["L"]:
            msg = "ell should be less than or equal to L"
            raise ValueError(msg)
        return v

    @pydantic.field_validator("m")
    def _check_m(
        cls,  # noqa: ANN101
        v: int,
        info: pydantic.ValidationInfo,
    ) -> int:
        if not isinstance(v, int):
            msg = "m should be an integer"
            raise TypeError(msg)
        if abs(v) > info.data["ell"]:
            msg = "the magnitude of m should be less than ell"
            raise ValueError(msg)
        return v
