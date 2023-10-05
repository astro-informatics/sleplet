"""Contains the abstract `Flm` class."""
import abc

import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import pyssht as ssht
import s2fft

import sleplet._validation
import sleplet.noise
from sleplet.functions.coefficients import Coefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class Flm(Coefficients):
    """Abstract parent class to handle harmonic coefficients on the sphere."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def rotate(  # noqa: D102
        self,
        alpha: float,
        beta: float,
        *,
        gamma: float = 0,
    ) -> npt.NDArray[np.complex_]:
        return s2fft.samples.flm_1d_to_2d(
            ssht.rotate_flms(
                s2fft.samples.flm_2d_to_1d(self.coefficients, self.L),
                alpha,
                beta,
                gamma,
                self.L,
            ),
            self.L,
        )

    def _translation_helper(
        self,
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.complex_]:
        return s2fft.samples.flm_1d_to_2d(
            ssht.create_ylm(beta, alpha, self.L).conj().flatten(),
            self.L,
        )

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise to the signal."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = sleplet.noise._create_noise(self.L, self.coefficients, self.noise)
            snr = sleplet.noise.compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
            return unnoised_coefficients, snr
        return None, None

    @abc.abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_reality(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_spin(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
