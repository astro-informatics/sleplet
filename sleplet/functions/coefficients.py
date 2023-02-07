from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sleplet.utils.convolution_methods import sifting_convolution
from sleplet.utils.mask_methods import ensure_masked_flm_bandlimited
from sleplet.utils.region import Region
from sleplet.utils.string_methods import filename_args

COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian", "south", "america"}


@dataclass
class Coefficients:
    L: int
    extra_args: Optional[list[int]]
    region: Optional[Region]
    noise: Optional[float]
    smoothing: Optional[int]
    _coefficients: np.ndarray = field(init=False, repr=False)
    _extra_args: Optional[list[int]] = field(default=None, init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _reality: bool = field(default=False, init=False, repr=False)
    _region: Optional[Region] = field(default=None, init=False, repr=False)
    _noise: Optional[float] = field(default=None, init=False, repr=False)
    _smoothing: Optional[int] = field(default=None, init=False, repr=False)
    _spin: int = field(default=0, init=False, repr=False)
    _unnoised_coefficients: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._setup_args()
        self._create_name()
        self._set_spin()
        self._set_reality()
        self._create_coefficients()
        self._add_details_to_name()
        self._add_noise_to_signal()

    def translate(
        self, alpha: float, beta: float, *, shannon: Optional[int] = None
    ) -> np.ndarray:
        g_coefficients = self._translation_helper(alpha, beta)
        return (
            g_coefficients
            if "dirac_delta" in self.name
            else self.convolve(self.coefficients, g_coefficients, shannon=shannon)
        )

    def convolve(
        self,
        f_coefficient: np.ndarray,
        g_coefficient: np.ndarray,
        *,
        shannon: Optional[int] = None,
    ) -> np.ndarray:
        # translation/convolution are not real for general function
        self.reality = False
        return sifting_convolution(f_coefficient, g_coefficient, shannon=shannon)

    def _add_details_to_name(self) -> None:
        """
        adds region to the name if present if not a Slepian function
        adds noise/smoothing if appropriate and bandlimit
        """
        if (
            isinstance(self.region, Region)
            and not set(self.name.split("_")) & COEFFICIENTS_TO_NOT_MASK
        ):
            self.name += f"_{self.region.name_ending}"
        if self.noise is not None:
            self.name += f"{filename_args(self.noise, 'noise')}"
        if self.smoothing is not None:
            self.name += f"{filename_args(self.smoothing, 'smoothed')}"
        self.name += f"_L{self.L}"

    @coefficients.setter
    def coefficients(self, coefficients: np.ndarray) -> None:
        if (
            isinstance(self.region, Region)
            and not set(self.name.split("_")) & COEFFICIENTS_TO_NOT_MASK
        ):
            coefficients = ensure_masked_flm_bandlimited(
                coefficients, self.L, self.region, self.reality, self.spin
            )
        self._coefficients = coefficients

    @extra_args.setter
    def extra_args(self, extra_args: Optional[list[int]]) -> None:
        self._extra_args = extra_args

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @noise.setter
    def noise(self, noise: Optional[float]) -> None:
        self._noise = noise

    @reality.setter
    def reality(self, reality: bool) -> None:
        self._reality = reality

    @region.setter
    def region(self, region: Optional[Region]) -> None:
        self._region = region

    @smoothing.setter
    def smoothing(self, smoothing: Optional[int]) -> None:
        self._smoothing = smoothing

    @spin.setter
    def spin(self, spin: int) -> None:
        self._spin = spin

    @unnoised_coefficients.setter
    def unnoised_coefficients(
        self, unnoised_coefficients: Optional[np.ndarray]
    ) -> None:
        self._unnoised_coefficients = unnoised_coefficients

    @abstractmethod
    def rotate(self, alpha: float, beta: float, *, gamma: float = 0) -> np.ndarray:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        raise NotImplementedError

    @abstractmethod
    def _translation_helper(self, alpha: float, beta: float) -> np.ndarray:
        """
        compute the basis function at omega' for translation
        """
        raise NotImplementedError

    @abstractmethod
    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> None:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the function
        """
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> None:
        """
        sets the reality flag to speed up computations
        """
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> None:
        """
        sets the spin value in computations
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        """
        initialises function specific args
        either default value or user input
        """
        raise NotImplementedError
