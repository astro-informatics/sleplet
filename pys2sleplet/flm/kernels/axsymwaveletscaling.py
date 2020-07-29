from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.pys2let import s2let
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class AxsymWaveletScaling(Functions):
    B: int
    j_min: int
    _B: int = field(default=3, init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        kappa0, _ = s2let.axisym_wav_l(self.B, self.L, self.j_min)
        flm = np.zeros(self.L ** 2, dtype=np.complex128)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = kappa0[ell]
        self.multipole = flm

    def _create_name(self) -> None:
        self.name = (
            "axsym_wavelet_scaling"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min = self.extra_args

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = AxsymWaveletScaling._B
        self._B = B

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = AxsymWaveletScaling._j_min
        self._j_min = j_min
