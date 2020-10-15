from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class Gaussian(F_LM):
    sigma: float
    _sigma: float = field(default=1_000, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_coefficients(self) -> None:
        flm = np.zeros(self.L ** 2, dtype=np.complex128)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma ** 2))
        self.coefficients = flm

    def _create_name(self) -> None:
        self.name = f"gaussian{filename_args(self.sigma, 'sig')}"

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.sigma = 10 ** self.extra_args[0]

    @property  # type:ignore
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if isinstance(sigma, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            sigma = Gaussian._sigma
        self._sigma = sigma
