from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.utils.convolution_methods import sifting_convolution
from pys2sleplet.utils.mask_methods import ensure_masked_flm_bandlimited
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.string_methods import filename_args

COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian", "south", "america"}


@dataclass  # type:ignore
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

    def __post_init__(self) -> None:
        self._setup_args()
        self._create_name()
        self._set_spin()
        self._set_reality()
        self._create_coefficients()
        self._add_details_to_name()
        self._add_noise_to_signal()

    def translate(
        self, alpha: float, beta: float, shannon: Optional[int] = None
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

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

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

    @property  # type:ignore
    def extra_args(self) -> Optional[list[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[list[int]]) -> None:
        if isinstance(extra_args, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            extra_args = Coefficients._extra_args
        self._extra_args = extra_args

    @property  # type:ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type:ignore
    def noise(self) -> Optional[float]:
        return self._noise

    @noise.setter
    def noise(self, noise: Optional[float]) -> None:
        if isinstance(noise, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            noise = Coefficients._noise
        self._noise = noise

    @property
    def reality(self) -> bool:
        return self._reality

    @reality.setter
    def reality(self, reality: bool) -> None:
        self._reality = reality

    @property  # type:ignore
    def region(self) -> Optional[Region]:
        return self._region

    @region.setter
    def region(self, region: Optional[Region]) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = Coefficients._region
        self._region = region

    @property  # type:ignore
    def smoothing(self) -> Optional[int]:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing: Optional[int]) -> None:
        if isinstance(smoothing, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            smoothing = Coefficients._smoothing
        self._smoothing = smoothing

    @property
    def spin(self) -> int:
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        if isinstance(spin, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            spin = Coefficients._spin
        self._spin = spin

    @abstractmethod
    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
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
