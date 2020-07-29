from dataclasses import dataclass, field

import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.harmonic_methods import create_spherical_harmonic
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SphericalHarmonic(Functions):
    ell: int
    m: int
    _ell: int = field(default=0, init=False, repr=False)
    _m: int = field(default=0, init=False, repr=False)

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        ind = ssht.elm2ind(self.ell, self.m)
        self.multipole = create_spherical_harmonic(self.L, ind)

    def _create_name(self) -> None:
        self.name = (
            "spherical_harmonic"
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

    @property  # type:ignore
    def ell(self) -> int:
        return self._ell

    @ell.setter
    def ell(self, ell: int) -> None:
        if isinstance(ell, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            ell = SphericalHarmonic._ell
        if not isinstance(ell, int):
            raise TypeError("ell should be an integer")
        if ell < 0:
            raise ValueError("ell should be positive")
        if ell >= self.L:
            raise ValueError("ell should be less than or equal to L")
        self._ell = ell

    @property  # type:ignore
    def m(self) -> int:
        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if isinstance(m, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            m = SphericalHarmonic._m
        if not isinstance(m, int):
            raise TypeError("m should be an integer")
        if abs(m) > self.ell:
            raise ValueError("the magnitude of m should be less than ell")
        self._m = m
