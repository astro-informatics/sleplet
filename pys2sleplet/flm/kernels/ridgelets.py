from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyssht as ssht
from scipy.special import gammaln

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.pys2let import s2let
from pys2sleplet.utils.string_methods import filename_args, wavelet_ending


@dataclass
class Ridgelets(Functions):
    B: int
    j_min: int
    spin: int
    j: Optional[int]
    _B: int = field(default=2, init=False, repr=False)
    _j_min: int = field(default=3, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _spin: int = field(default=2, init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        logger.info("start computing wavelets")
        self._create_wavelets()
        logger.info("finish computing wavelets")
        self.multipole = (
            self.wavelets[0] if self.j is None else self.wavelets[self.j + 1]
        )

    def _create_name(self) -> None:
        self.name = (
            "ridgelets"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{filename_args(self.spin, 'spin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 4
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.j = self.extra_args

    def _create_wavelets(self) -> None:
        """
        compute all wavelets
        """
        ring_lm = self._compute_ring()
        kappa0, kappa = s2let.axisym_wav_l(self.B, self.L, self.j_min)
        self.wavelets = np.zeros((kappa.shape[1] + 1, self.L ** 2), dtype=np.complex128)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            self.wavelets[0, ind] = kappa0[ell] * ring_lm[ind]
            self.wavelets[1:, ind] = kappa[ell] * ring_lm[ind] / np.sqrt(2 * np.pi)

    def _compute_ring(self) -> np.ndarray:
        """
        compute ring in harmonic space
        """
        ring_lm = np.zeros(self.L ** 2, dtype=np.complex128)
        for ell in range(abs(self.spin), self.L):
            logp2 = (
                gammaln(ell + self.spin + 1)
                - ell * np.log(2)
                - gammaln((ell + self.spin) / 2 + 1)
                - gammaln((ell - self.spin) / 2 + 1)
            )
            p0 = np.real((-1) ** ((ell + self.spin) / 2)) * np.exp(logp2)
            ind = ssht.elm2ind(ell, 0)
            ring_lm[ind] = (
                2
                * np.pi
                * np.sqrt((2 * ell + 1) / (4 * np.pi))
                * p0
                * (-1) ** self.spin
                * np.sqrt(
                    np.exp(gammaln(ell - self.spin + 1) - gammaln(ell + self.spin + 1))
                )
            )
        return ring_lm

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = Ridgelets._B
        self._B = B

    @property  # type:ignore
    def j(self) -> Optional[int]:
        return self._j

    @j.setter
    def j(self, j: Optional[int]) -> None:
        if isinstance(j, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j = Ridgelets._j
        j_max = s2let.pys2let_j_max(self.B, self.L, self.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - self.j_min + 1}"
            )
        self._j = j

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = Ridgelets._j_min
        self._j_min = j_min

    @property  # type:ignore
    def spin(self) -> int:
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        if isinstance(spin, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            spin = Ridgelets._spin
        self._spin = spin

    @property  # type:ignore
    def wavelets(self) -> np.ndarray:
        return self._wavelets

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets
