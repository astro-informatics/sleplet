from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pys2let as s2let
import pyssht as ssht

from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.string_methods import filename_args, wavelet_ending


@dataclass
class DirectionalSpinWavelets(F_LM):
    B: int
    j_min: int
    spin: int
    N: int
    j: Optional[int]
    _B: int = field(default=2, init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _N: int = field(default=2, init=False, repr=False)
    _spin: int = field(default=0, init=False, repr=False)
    _j_max: int = field(init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelets")
        self._create_wavelets()
        logger.info("finish computing wavelets")
        self.coefficients = (
            self.wavelets[0] if self.j is None else self.wavelets[self.j + 1]
        )

    def _create_name(self) -> None:
        self.name = (
            "directional_spin_wavelets"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{filename_args(self.spin, 'spin')}"
            f"{filename_args(self.N, 'N')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> None:
        self.reality = self.j is None or self.spin == 0

    def _set_spin(self) -> None:
        self.spin = self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 5
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.N, self.j = self.extra_args

    def _create_wavelets(self) -> np.ndarray:
        """
        compute all wavelets
        """
        phi_l, psi_lm = s2let.wavelet_tiling(
            self.B, self.L, self.N, self.j_min, self.spin
        )
        self.wavelets = np.zeros(
            (psi_lm.shape[1] + 1, self.L ** 2), dtype=np.complex128
        )
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            self.wavelets[0, ind] = phi_l[ell]
        self.wavelets[1:] = psi_lm.T

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = DirectionalSpinWavelets._B
        self._B = B

    @property  # type:ignore
    def j(self) -> Optional[int]:
        return self._j

    @j.setter
    def j(self, j: Optional[int]) -> None:
        if isinstance(j, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j = DirectionalSpinWavelets._j
        self.j_max = s2let.pys2let_j_max(self.B, self.L, self.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > self.j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {self.j_max - self.j_min + 1}"
            )
        self._j = j

    @property  # type:ignore
    def j_max(self) -> int:
        return self._j_max

    @j_max.setter
    def j_max(self, j_max: int) -> None:
        self._j_max = j_max

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = DirectionalSpinWavelets._j_min
        self._j_min = j_min

    @property  # type:ignore
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        if isinstance(N, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            N = DirectionalSpinWavelets._N
        self._N = N

    @property  # type:ignore
    def spin(self) -> int:
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        if isinstance(spin, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            spin = DirectionalSpinWavelets._spin
        self._spin = spin

    @property  # type:ignore
    def wavelets(self) -> np.ndarray:
        return self._wavelets

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets
