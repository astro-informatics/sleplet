"""Contains the abstract `Coefficients` class."""
import abc
import dataclasses

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._convolution_methods
import sleplet._mask_methods
import sleplet._string_methods
import sleplet._validation
import sleplet.slepian.region

_COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian", "south", "america"}


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Coefficients:
    """
    Abstract parent class to handle harmonic/Slepian coefficients on the
    sphere.
    """

    L: int
    """The spherical harmonic bandlimit."""
    _: dataclasses.KW_ONLY
    extra_args: list[int] | None = None
    """Control the extra arguments for the given set of spherical
    coefficients. Only to be set by the `sphere` CLI."""
    noise: float | None = None
    """How much to noise the data."""
    region: sleplet.slepian.region.Region | None = None
    """Whether to set a region or not, used in the Slepian case."""
    smoothing: int | None = None
    """How much to smooth the topographic map of the Earth by."""
    _unnoised_coefficients: (
        npt.NDArray[np.complex_ | np.float_] | None
    ) = pydantic.Field(
        default=None,
        init_var=False,
        repr=False,
    )
    coefficients: npt.NDArray[np.complex_ | np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    name: str = pydantic.Field(default="", init_var=False, repr=False)
    reality: bool = pydantic.Field(default=False, init_var=False, repr=False)
    snr: float | None = pydantic.Field(default=None, init_var=False, repr=False)
    spin: int = pydantic.Field(default=0, init_var=False, repr=False)
    wavelet_coefficients: npt.NDArray[np.complex_ | np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    wavelets: npt.NDArray[np.complex_ | np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self._setup_args()
        self.name = self._create_name()
        self.spin = self._set_spin()
        self.reality = self._set_reality()
        self.coefficients = self._create_coefficients()
        self._add_details_to_name()
        self._unnoised_coefficients, self.snr = self._add_noise_to_signal()

    def translate(
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
        *,
        shannon: int | None = None,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        r"""
        Perform the translation of the coefficients, used in the sifting convolution.

        Args:
            alpha: The point on the 2-sphere to translate to, i.e. the \(\phi\) value.
            beta: The point on the 2-sphere to translate to, i.e. the \(\theta\) value.
            shannon: The Shannon number, only used in the Slepian case.

        Returns:
            The translated spherical harmonic coefficients.
        """
        g_coefficients = self._translation_helper(alpha, beta)
        return (
            g_coefficients
            if "dirac_delta" in self.name
            else self.convolve(self.coefficients, g_coefficients, shannon=shannon)
        )

    def convolve(
        self: typing_extensions.Self,
        f_coefficient: npt.NDArray[np.complex_ | np.float_],
        g_coefficient: npt.NDArray[np.complex_ | np.float_],
        *,
        shannon: int | None = None,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        """
        Perform the sifting convolution of the two inputs.

        Args:
            f_coefficient: Input harmonic/Slepian coefficients.
            g_coefficient: Input harmonic/Slepian coefficients.
            shannon: The Shannon number, only used in the Slepian case.

        Returns:
            The sifting convolution of the two inputs.
        """
        # translation/convolution are not real for general function
        self.reality = False
        return sleplet._convolution_methods.sifting_convolution(
            f_coefficient,
            g_coefficient,
            shannon=shannon,
        )

    def _add_details_to_name(self: typing_extensions.Self) -> None:
        """
        Add region to the name if present if not a Slepian function
        adds noise/smoothing if appropriate and bandlimit.
        """
        if (
            isinstance(self.region, sleplet.slepian.region.Region)
            and not set(self.name.split("_")) & _COEFFICIENTS_TO_NOT_MASK
        ):
            self.name += f"_{self.region._name_ending}"
        if self.noise is not None:
            self.name += f"{sleplet._string_methods.filename_args(self.noise, 'noise')}"
        if self.smoothing is not None:
            self.name += (
                f"{sleplet._string_methods.filename_args(self.smoothing, 'smoothed')}"
            )
        self.name += f"_L{self.L}"

    @abc.abstractmethod
    def rotate(
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
        *,
        gamma: float = 0,
    ) -> npt.NDArray[np.complex_]:
        r"""
        Rotates given flm on the sphere by alpha/beta/gamma.

        Args:
            alpha: The third Euler angle, a \(\alpha\) rotation about the z-axis.
            beta: The second Euler angle, a \(\beta\) rotation about the y-axis.
            gamma: The first Euler angle, a \(\gamma\) rotation about the z-axis.

        Returns:
            The rotated spherical harmonic coefficients.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _translation_helper(
        self: typing_extensions.Self,
        alpha: float,
        beta: float,
    ) -> npt.NDArray[np.complex_]:
        """Compute the basis function at omega' for translation."""
        raise NotImplementedError

    @abc.abstractmethod
    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise to the signal."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        """Create the flm on the north pole."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self: typing_extensions.Self) -> str:
        """Create the name of the function."""
        raise NotImplementedError

    @abc.abstractmethod
    def _set_reality(self: typing_extensions.Self) -> bool:
        """Set the reality flag to speed up computations."""
        raise NotImplementedError

    @abc.abstractmethod
    def _set_spin(self: typing_extensions.Self) -> int:
        """Set the spin value in computations."""
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self: typing_extensions.Self) -> None:
        """Initialise function specific args either default value or user input."""
        raise NotImplementedError
