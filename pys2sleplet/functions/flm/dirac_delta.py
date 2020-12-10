from dataclasses import dataclass

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.f_lm import F_LM


@dataclass
class DiracDelta(F_LM):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_coefficients(self) -> None:
        flm = np.zeros(self.L ** 2, dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        self.coefficients = flm

    def _create_name(self) -> None:
        self.name = "dirac_delta"

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
