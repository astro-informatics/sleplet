import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet import logger
from sleplet.functions.f_p import F_P
from sleplet.utils._validation import Validation
from sleplet.utils._vars import SAMPLING_SCHEME
from sleplet.utils.slepian_methods import _compute_s_p_omega_prime
from sleplet.utils.string_methods import _convert_camel_case_to_snake_case


@dataclass(config=Validation)
class SlepianDiracDelta(F_P):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        self._compute_angles()
        return _compute_s_p_omega_prime(
            self.L, self.alpha, self.beta, self.slepian
        ).conj()

    def _create_name(self) -> str:
        return (
            f"{_convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )

    def _compute_angles(self) -> None:
        """
        computes alpha/beta if not provided
        """
        thetas, phis = ssht.sample_positions(self.L, Grid=True, Method=SAMPLING_SCHEME)
        sp = ssht.inverse(self.slepian.eigenvectors[0], self.L, Method=SAMPLING_SCHEME)
        idx = tuple(np.argwhere(sp == sp.max())[0])
        self.alpha = phis[idx]
        self.beta = thetas[idx]
        logger.info(
            f"angles: (alpha, beta) = ({self.alpha/np.pi:.5f},{self.beta/np.pi:.5f})"
        )
        logger.info(f"grid point: (alpha, beta) = ({self.alpha:e},{self.beta:e})")
