from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.pys2let import s2let
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class AxsymWaveletCoefficients(Functions):
    B: int
    j_min: int
    j: int
    _B: int = field(default=3, init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)
    _j: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        _, kappa = s2let.axisym_wav_l(self.B, self.L, self.j_min)
        flm = np.zeros(self.L ** 2, dtype=np.complex128)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = kappa[ell, self.j]
        self.multipole = flm

    def _create_name(self) -> None:
        self.name = (
            "axsym_wavelet_coefficients"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{filename_args(self.j + self.j_min, 'j')}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = AxsymWaveletCoefficients._B
        self._B = B

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = AxsymWaveletCoefficients._j_min
        self._j_min = j_min

    @property  # type:ignore
    def j(self) -> int:
        return self._j

    @j.setter
    def j(self, j: int) -> None:
        if isinstance(j, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j = AxsymWaveletCoefficients._j
        j_max = s2let.pys2let_j_max(self.B, self.L, self.j_min)
        if j < 0:
            raise ValueError("j should be positive")
        if j > j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - self.j_min + 1}"
            )
        self._j = j
