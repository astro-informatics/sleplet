"""Contains the abstract `Fp` class."""
import abc

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._mask_methods
import sleplet._validation
import sleplet.noise
import sleplet.slepian.region
import sleplet.slepian.slepian_functions
import sleplet.slepian.slepian_polar_cap
import sleplet.slepian_methods
from sleplet.functions.coefficients import Coefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Fp(Coefficients):
    """Abstract parent class to handle Slepian coefficients on the sphere."""

    slepian: sleplet.slepian.slepian_functions.SlepianFunctions | None = pydantic.Field(
        default=None,
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self.region: sleplet.slepian.region.Region | None = (
            self.region
            if isinstance(self.region, sleplet.slepian.region.Region)
            else sleplet._mask_methods.create_default_region()
        )
        self.slepian = sleplet.slepian_methods.choose_slepian_method(
            self.L,
            self.region,
        )
        super().__post_init__()

    def rotate(  # noqa: D102
        self: typing_extensions.Self,
        alpha: float,  # noqa: ARG002
        beta: float,  # noqa: ARG002
        *,
        gamma: float = 0,  # noqa: ARG002
    ) -> npt.NDArray[np.complex_]:
        msg = "Slepian rotation is not defined"
        raise NotImplementedError(msg)

    def _translation_helper(
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.complex_]:
        return sleplet.slepian_methods._compute_s_p_omega_prime(
            self.L,
            alpha,
            beta,
            self.slepian,
        ).conj()

    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise converted to Slepian space."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            n_p = sleplet.noise._create_slepian_noise(
                self.L,
                self.coefficients,
                self.slepian,
                self.noise,
            )
            snr = sleplet.noise.compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients = self.coefficients + n_p
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
