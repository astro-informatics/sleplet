import pyssht as ssht
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.functions.f_lm import F_LM
from sleplet.utils.harmonic_methods import create_spherical_harmonic
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args


@dataclass(kw_only=True)
class SphericalHarmonic(F_LM):
    ell: int = 0
    m: int = 0

    def _create_coefficients(self) -> None:
        ind = ssht.elm2ind(self.ell, self.m)
        self.coefficients = create_spherical_harmonic(self.L, ind)

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.ell, 'l')}"
            f"{filename_args(self.m, 'm')}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.ell, self.m = self.extra_args

    @validator("ell")
    def check_ell(cls, ell: int) -> int:
        if not isinstance(ell, int):
            raise TypeError("ell should be an integer")
        if ell < 0:
            raise ValueError("ell should be positive")
        if ell >= cls.L:
            raise ValueError("ell should be less than or equal to L")
        return ell

    @validator("m")
    def check_m(cls, m: int) -> int:
        if not isinstance(m, int):
            raise TypeError("m should be an integer")
        if abs(m) > cls.ell:
            raise ValueError("the magnitude of m should be less than ell")
        return m
