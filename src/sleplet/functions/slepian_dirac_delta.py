"""Contains the `SlepianDiracDelta` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import s2fft

import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.slepian_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class SlepianDiracDelta(Fp):
    """Creates a Dirac delta of the Slepian coefficients."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        self._compute_angles()
        return sleplet.slepian_methods._compute_s_p_omega_prime(
            self.L,
            self._alpha,
            self._beta,
            self.slepian,
        ).conj()

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )

    def _compute_angles(self) -> None:
        """Computes alpha/beta if not provided."""
        thetas = np.tile(
            s2fft.samples.thetas(self.L, sampling=sleplet._vars.SAMPLING_SCHEME),
            (
                s2fft.samples.nphi_equiang(
                    self.L,
                    sampling=sleplet._vars.SAMPLING_SCHEME,
                ),
                1,
            ),
        ).T
        phis = np.tile(
            s2fft.samples.phis_equiang(self.L, sampling=sleplet._vars.SAMPLING_SCHEME),
            (s2fft.samples.ntheta(self.L, sampling=sleplet._vars.SAMPLING_SCHEME), 1),
        )
        sp = s2fft.inverse(
            s2fft.sampling.s2_samples.flm_1d_to_2d(
                self.slepian.eigenvectors[0],
                self.L,
            ),
            self.L,
            method=sleplet._vars.EXECUTION_MODE,
            sampling=sleplet._vars.SAMPLING_SCHEME,
        )
        idx = tuple(np.argwhere(sp == sp.max())[0])
        self._alpha = phis[idx]
        self._beta = thetas[idx]
        _logger.info(
            f"angles: (alpha, beta) = ({self._alpha/np.pi:.5f},{self._beta/np.pi:.5f})",
        )
        _logger.info(
            f"grid point: (alpha, beta) = ({self._alpha:e},{self._beta:e})",
        )
