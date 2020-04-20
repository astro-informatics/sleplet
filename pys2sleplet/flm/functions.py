from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.plot_methods import calc_nearest_grid_point, calc_resolution
from pys2sleplet.utils.string_methods import filename_angle


@dataclass  # type: ignore
class Functions:
    L: int
    extra_args: Optional[List[int]] = field(default=None)
    __L: int = field(init=False, repr=False)
    __annotations: List[Dict] = field(init=False, repr=False)
    __field: np.ndarray = field(init=False, repr=False)
    __multipole: np.ndarray = field(init=False, repr=False)
    __name: str = field(init=False, repr=False)
    __plot: np.ndarray = field(init=False, repr=False)
    __resolution: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._setup_args(self.extra_args)
        self.resolution = calc_resolution(self.L)
        self.name = self._create_name()
        self.multipole = self._create_flm(self.L)
        self.field = self._invert(self.multipole)
        self.plot = self._invert(self.multipole, boosted=True)
        self.annotations = self._create_annotations()

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
            Path(__file__).resolve().parents[1]
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

    def _boost_res(self, flm) -> np.ndarray:
        """
        calculates a boost in resolution for given flm
        """
        boost = self.resolution * self.resolution - self.L * self.L
        flm_boost = np.pad(self.multipole, (0, boost), "constant")
        return flm_boost

    def _invert(self, flm: np.ndarray, boosted: bool = False) -> np.ndarray:
        """
        performs the inverse harmonic transform
        """
        # boost resolution for plot
        if boosted:
            flm = self._boost_res(flm)
            bandlimit = self.resolution
        else:
            bandlimit = self.L

        # perform inverse
        f = ssht.inverse(flm, bandlimit, Reality=self.reality, Method="MWSS")
        return f

    @property  # type: ignore
    def L(self) -> int:
        return self.__L

    @L.setter
    def L(self, var: int) -> None:
        self.__L = var

    @property
    def resolution(self) -> int:
        return self.__resolution

    @resolution.setter
    def resolution(self, var: int) -> None:
        self.__resolution = var

    @property
    def reality(self) -> bool:
        return self.__reality

    @reality.setter
    def reality(self, var: bool) -> None:
        self.__reality = var

    @property
    def multipole(self) -> np.ndarray:
        return self.__multipole

    @multipole.setter
    def multipole(self, var: np.ndarray) -> None:
        """
        update multipole value and hence field value
        """
        self.__multipole = var
        self.field = self._invert(self.multipole)
        self.plot = self._invert(self.multipole, boosted=True)

    @property
    def name(self) -> np.ndarray:
        return self.__name

    @name.setter
    def name(self, var: str) -> None:
        self.__name = var

    @property
    def field(self) -> np.ndarray:
        return self.__field

    @field.setter
    def field(self, var: np.ndarray) -> None:
        self.__field = var

    @property
    def plot(self) -> np.ndarray:
        return self.__plot

    @plot.setter
    def plot(self, var: np.ndarray) -> None:
        self.__plot = var

    @abstractmethod
    def _setup_args(self, args: Optional[List[int]]) -> None:
        """
        initialises function specific args
        either default value or user input
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
