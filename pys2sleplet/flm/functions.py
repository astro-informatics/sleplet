from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.plot_methods import calc_nearest_grid_point, calc_resolution
from pys2sleplet.utils.string_methods import filename_angle


class Functions:
    def __init__(self, L: int, extra_args: Optional[List[int]]):
        self.L = L
        self.resolution = calc_resolution(self.L)
        self._setup_args(extra_args)
        self.name = self._create_name()
        self.multipole = self._create_flm(self.L)
        self.field = self._invert(self.multipole)

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

    @property
    def L(self) -> int:
        return self.__L

    @L.setter
    def L(self, var: int) -> None:
        """
        update L and hence resolution
        """
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
            glm = np.conj(ssht.create_ylm(beta, alpha, self.L))
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

        self.multipole *= np.conj(glm)

    def _boost_res(self, flm) -> np.ndarray:
        """
        calculates a boost in resolution for given flm
        """
        boost = self.resolution * self.resolution - self.L * self.L
        flm_boost = np.pad(self.multipole, (0, boost), "constant")
        return flm_boost

    def _invert(self, flm: np.ndarray) -> np.ndarray:
        """
        performs the inverse harmonic transform
        """
        # boost resolution for plot
        flm_boost = self._boost_res(flm)

        # perform inverse
        f = ssht.inverse(
            flm_boost, self.resolution, Reality=self.reality, Method="MWSS"
        )
        return f
