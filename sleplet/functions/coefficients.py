from abc import abstractmethod
from dataclasses import KW_ONLY, field
from typing import Optional

import numpy as np
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.convolution_methods import sifting_convolution
from sleplet.utils.mask_methods import ensure_masked_flm_bandlimited
from sleplet.utils.region import Region
from sleplet.utils.string_methods import filename_args

COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian", "south", "america"}


@dataclass
class Coefficients:
    L: int
    _: KW_ONLY
    coefficients: np.ndarray = field(init=False, repr=False)
    extra_args: Optional[list[int]] = None
    name: str = field(init=False, repr=False)
    noise: Optional[int] = None
    region: Optional[Region] = None
    slepian: SlepianFunctions = field(init=False, repr=False)
    smoothing: Optional[int] = None
    snr: float = field(init=False, repr=False)
    spin: int = field(init=False, repr=False)
    unnoised_coefficients: np.ndarray = field(init=False, repr=False)

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

    @validator("coefficients")
    def check_coefficients(cls, coefficients: np.ndarray) -> np.ndarray:
        if (
            isinstance(cls.region, Region)
            and not set(cls.name.split("_")) & COEFFICIENTS_TO_NOT_MASK
        ):
            coefficients = ensure_masked_flm_bandlimited(
                coefficients, cls.L, cls.region, cls.reality, cls.spin
            )
        return coefficients

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
