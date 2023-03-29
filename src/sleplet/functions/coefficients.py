from abc import abstractmethod
from dataclasses import KW_ONLY

import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.utils._convolution_methods import sifting_convolution
from sleplet.utils._mask_methods import ensure_masked_flm_bandlimited
from sleplet.utils._validation import Validation
from sleplet.utils.region import Region
from sleplet.utils.string_methods import filename_args

COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian", "south", "america"}


@dataclass(config=Validation)
class Coefficients:
    L: int
    _: KW_ONLY
    extra_args: list[int] | None = None
    noise: float | None = None
    region: Region | None = None
    smoothing: int | None = None

    def __post_init_post_parse__(self) -> None:
        self._setup_args()
        self.name = self._create_name()
        self.spin = self._set_spin()
        self.reality = self._set_reality()
        self.coefficients = self._create_coefficients()
        self._add_details_to_name()
        self.unnoised_coefficients, self.snr = self._add_noise_to_signal()

    def translate(
        self, alpha: float, beta: float, *, shannon: int | None = None
    ) -> npt.NDArray[np.complex_ | np.float_]:
        g_coefficients = self._translation_helper(alpha, beta)
        return (
            g_coefficients
            if "dirac_delta" in self.name
            else self.convolve(self.coefficients, g_coefficients, shannon=shannon)
        )

    def convolve(
        self,
        f_coefficient: npt.NDArray[np.complex_ | np.float_],
        g_coefficient: npt.NDArray[np.complex_ | np.float_],
        *,
        shannon: int | None = None,
    ) -> npt.NDArray[np.complex_ | np.float_]:
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

    @validator("coefficients", check_fields=False)
    def _check_coefficients(cls, v, values):
        if (
            values["region"]
            and not set(values["name"].split("_")) & COEFFICIENTS_TO_NOT_MASK
        ):
            v = ensure_masked_flm_bandlimited(
                v,
                values["L"],
                values["region"],
                reality=values["reality"],
                spin=values["spin"],
            )
        return v

    @abstractmethod
    def rotate(
        self, alpha: float, beta: float, *, gamma: float = 0
    ) -> npt.NDArray[np.complex_]:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        raise NotImplementedError

    @abstractmethod
    def _translation_helper(
        self, alpha: float, beta: float
    ) -> npt.NDArray[np.complex_]:
        """
        compute the basis function at omega' for translation
        """
        raise NotImplementedError

    @abstractmethod
    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """
        adds Gaussian white noise to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        """
        creates the name of the function
        """
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> bool:
        """
        sets the reality flag to speed up computations
        """
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> int:
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
