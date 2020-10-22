from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import compute_s_p_omega_prime


@dataclass
class SlepianDiracDelta(F_P):
    _alpha: float = field(init=False, repr=False)
    _beta: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_coefficients(self) -> None:
        self._compute_angles()
        self.coefficients = compute_s_p_omega_prime(
            self.L, self.alpha, self.beta, self.slepian
        ).conj()

    def _create_name(self) -> None:
        self.name = f"slepian_dirac_delta_{self.slepian.region.name_ending}"

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )

    def _compute_angles(self) -> None:
        """
        computes alpha/beta if not provided
        """
        thetas, phis = ssht.sample_positions(self.L, Grid=True)
        sp = ssht.inverse(self.slepian.eigenvectors[0], self.L)
        idx = tuple(np.argwhere(sp == sp.max())[0])
        self.alpha = phis[idx]
        self.beta = thetas[idx]
        logger.info(
            f"{self.name} grid point: (alpha, beta) = ({self.alpha:e},{self.beta:e})"
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        self._beta = beta
