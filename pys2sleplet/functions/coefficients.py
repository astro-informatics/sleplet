from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.utils.convolution_methods import sifting_convolution
from pys2sleplet.utils.mask_methods import ensure_masked_flm_bandlimited
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region

_file_location = Path(__file__).resolve()


@dataclass  # type:ignore
class Coefficients:
    L: int
    extra_args: Optional[List[int]]
    region: Optional[Region]
    noise: int
    smoothing: int
    _annotations: List[Dict] = field(default_factory=list, init=False, repr=False)
    _coefficients: np.ndarray = field(init=False, repr=False)
    _extra_args: Optional[List[int]] = field(default=None, init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _reality: bool = field(default=False, init=False, repr=False)
    _region: Region = field(default=None, init=False, repr=False)
    _noise: int = field(default=0, init=False, repr=False)
    _resolution: int = field(init=False, repr=False)
    _smoothing: int = field(default=0, init=False, repr=False)
    _spin: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.resolution = calc_plot_resolution(self.L)
        self._setup_args()
        self._create_name()
        self._create_annotations()
        self._set_spin()
        self._set_reality()
        self._create_coefficients()
        self._add_region_to_name()
        self._add_noise_to_signal()
        self._smooth_signal()

    def convolve(
        self,
        f_coefficient: np.ndarray,
        g_coefficient: np.ndarray,
        shannon: Optional[int] = None,
    ) -> np.ndarray:
        # translation/convolution are not real for general function
        self.reality = False
        return sifting_convolution(f_coefficient, g_coefficient, shannon=shannon)

    def _add_region_to_name(self) -> None:
        """
        adds region to the name if present if not a Slepian function
        """
        if self.region is not None and "slepian" not in self.name:
            self.name += f"_{self.region.name_ending}"

    @property
    def annotations(self) -> List[Dict]:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: List[Dict]) -> None:
        self._annotations = annotations

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients: np.ndarray) -> None:
        if self.region is not None and all(
            x not in self.name for x in {"slepian", "south_america"}
        ):
            coefficients = ensure_masked_flm_bandlimited(
                coefficients, self.L, self.region, self.reality, self.spin
            )
        self._coefficients = coefficients

    @property  # type:ignore
    def extra_args(self) -> Optional[List[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[List[int]]) -> None:
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
    def name(self) -> np.ndarray:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type:ignore
    def noise(self) -> int:
        return self._noise

    @noise.setter
    def noise(self, noise: int) -> None:
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
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = Coefficients._region
        self._region = region

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int) -> None:
        self._resolution = resolution

    @property  # type:ignore
    def smoothing(self) -> int:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing: int) -> None:
        if isinstance(smoothing, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            smoothing = Coefficients._smoothing
        self._smoothing = smoothing

    @property  # type:ignore
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
    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        """
        computes the inverse of the given coefficients
        """
        raise NotImplementedError

    @abstractmethod
    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        raise NotImplementedError

    @abstractmethod
    def translate(
        self, alpha: float, beta: float, shannon: Optional[int] = None
    ) -> np.ndarray:
        """
        translates given flm on the sphere by alpha/beta
        """
        raise NotImplementedError

    @abstractmethod
    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _smooth_signal(self) -> None:
        """
        applies Gaussian smoothing to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _create_annotations(self) -> None:
        """
        creates the annotations for the plot
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
