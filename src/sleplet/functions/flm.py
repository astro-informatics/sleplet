"""Contains the abstract `Flm` class."""
import abc

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._validation
import sleplet.noise
from sleplet.functions.coefficients import Coefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Flm(Coefficients):
    """Abstract parent class to handle harmonic coefficients on the sphere."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def rotate(  # noqa: D102
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
        *,
        gamma: float = 0,
    ) -> npt.NDArray[np.complex_]:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def _translation_helper(
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.complex_]:
        return ssht.create_ylm(beta, alpha, self.L).conj().flatten()

    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise to the signal."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = sleplet.noise._create_noise(self.L, self.coefficients, self.noise)
            snr = sleplet.noise.compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
            return unnoised_coefficients, snr
        return None, None

    @abc.abstractmethod
    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self: typing_extensions.Self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_reality(self: typing_extensions.Self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_spin(self: typing_extensions.Self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self: typing_extensions.Self) -> None:
        raise NotImplementedError
