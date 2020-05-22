from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.config import config
from pys2sleplet.utils.harmonic_methods import invert_flm
from pys2sleplet.utils.plot_methods import calc_nearest_grid_point, calc_resolution
from pys2sleplet.utils.string_methods import filename_angle

_file_location = Path(__file__).resolve()


@dataclass  # type: ignore
class Functions:
    L: int
    extra_args: Optional[List[int]]
    _annotations: List[Dict] = field(default_factory=list, init=False, repr=False)
    _extra_args: Optional[List[int]] = field(default=None, init=False, repr=False)
    _field: np.ndarray = field(init=False, repr=False)
    _field_padded: np.ndarray = field(init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _multipole: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _reality: bool = field(default=False, init=False, repr=False)
    _resolution: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.annotations = self._create_annotations()
        self.name = self._create_name()
        self.reality = self._set_reality()

    def rotate(
        self,
        alpha_pi_fraction: float,
        beta_pi_fraction: float,
        gamma_pi_fraction: float = 0,
    ) -> None:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )
        gamma = gamma_pi_fraction * np.pi

        # rotate flms
        self.multipole = ssht.rotate_flms(self.multipole, alpha, beta, gamma, self.L)

    def translate(self, alpha_pi_fraction: float, beta_pi_fraction: float) -> None:
        """
        translates given flm on the sphere by alpha/beta
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )

        # numpy binary filename
        filename = (
            _file_location.parents[1]
            / "data"
            / "trans_dirac"
            / f"trans_dd_L{self.L}_{filename_angle(alpha_pi_fraction,beta_pi_fraction)}.npy"
        )

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        if filename.exists():
            glm = np.load(filename)
        else:
            glm = ssht.create_ylm(beta, alpha, self.L).conj()
            glm = glm.reshape(glm.size)

            # save to speed up for future
            if config.SAVE_MATRICES:
                np.save(filename, glm)

        # convolve with flm
        if self.name == "dirac_delta":
            self.multipole = glm
        else:
            self.convolve(glm)

    def convolve(self, glm: np.ndarray) -> None:
        """
        computes the sifting convolution of two arrays
        """
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        self.multipole *= glm.conj()

    @property
    def annotations(self) -> List[Dict]:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: List[Dict]) -> None:
        self._annotations = annotations

    @property  # type:ignore
    def extra_args(self) -> Optional[List[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[List[int]]) -> None:
        if isinstance(extra_args, property):
            # initial value not specified, use default
            extra_args = Functions._extra_args
        self._extra_args = extra_args
        self._setup_args(self.extra_args)
        self.multipole = self._create_flm(self.L)

    @property
    def field(self) -> np.ndarray:
        return self._field

    @field.setter
    def field(self, field: np.ndarray) -> None:
        self._field = field

    @property  # type: ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L
        self.resolution = calc_resolution(self.L)
        self.multipole = self._create_flm(self.L)

    @property
    def multipole(self) -> np.ndarray:
        return self._multipole

    @multipole.setter
    def multipole(self, multipole: np.ndarray) -> None:
        """
        update multipole value and hence field value
        """
        self._multipole = multipole
        self.field = invert_flm(self.multipole, self.L, reality=self.reality)
        self.field_padded = invert_flm(
            self.multipole, self.L, reality=self.reality, resolution=self.resolution
        )

    @property
    def name(self) -> np.ndarray:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def field_padded(self) -> np.ndarray:
        return self._field_padded

    @field_padded.setter
    def field_padded(self, field_padded: np.ndarray) -> None:
        self._field_padded = field_padded

    @property
    def reality(self) -> bool:
        return self._reality

    @reality.setter
    def reality(self, reality: bool) -> None:
        if isinstance(reality, property):
            # initial value not specified, use default
            reality = Functions._reality
        self._reality = reality

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int) -> None:
        self._resolution = resolution

    @abstractmethod
    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        """
        initialises function specific args
        either default value or user input
        """
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> bool:
        """
        sets the reality flag to speed up computations
        """
        raise NotImplementedError

    @abstractmethod
    def _create_flm(self, L: int) -> np.ndarray:
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
    def _create_annotations(self) -> List[Dict]:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError
