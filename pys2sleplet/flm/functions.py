from abc import abstractmethod
from pathlib import Path

import numpy as np
import pyssht as ssht

from .kernels.kernels import kernel_dict
from .maps.maps import map_dict
from pys2sleplet.utils.plot_methods import calc_nearest_grid_point, calc_resolution
from pys2sleplet.utils.string_methods import filename_angle


class Functions:
    def __init__(self, name: str, L: int):
        self.name = name
        self.L = L
        self.res = calc_resolution(L)
        self.__flm = None
        self.__reality = False

    @abstractmethod
    def _create_flm(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def flm(self) -> np.ndarray:
        return self.__flm

    @flm.setter
    def flm(self, var) -> None:
        self._create_flm()

    @property
    def reality(self) -> bool:
        return self.__reality

    @reality.setter
    def reality(self, var) -> None:
        self.__reality = var

    def rotate(
        self,
        alpha_pi_fraction: float,
        beta_pi_fraction: float,
        gamma_pi_fraction: float = 0,
    ) -> np.ndarray:
        """
        rotates given flm on the sphere by alpha/beta/gamma
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )
        gamma = gamma_pi_fraction * np.pi

        # rotate flms
        flm_rot = ssht.rotate_flms(self.flm, alpha, beta, gamma, self.L)
        return flm_rot

    def translate(
        self, alpha_pi_fraction: float, beta_pi_fraction: float
    ) -> np.ndarray:
        """
        translates given flm on the sphere by alpha/beta
        """
        # angles such that rotation and translation are equal
        alpha, beta = calc_nearest_grid_point(
            self.L, alpha_pi_fraction, beta_pi_fraction
        )

        # numpy binary filename
        filename = (
            Path(__file__).resolve().parent
            / "data"
            / "npy"
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
            flm_conv = glm
        else:
            flm_conv = self.convolve(glm)
        return flm_conv

    def convolve(self, glm: np.ndarray) -> np.ndarray:
        """
        computes the sifting convolution of two arrays
        """
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        return self.flm * np.conj(glm)

    def _boost_res(self) -> np.ndarray:
        """
        calculates a boost in resolution for given flm
        """
        boost = self.res * self.res - self.L * self.L
        flm_boost = np.pad(self.flm, (0, boost), "constant")
        return flm_boost

    def invert(self) -> np.ndarray:
        """
        """
        # boost resolution for plot
        flm = self._boost_res()

        # perform inverse
        f = ssht.inverse(flm, self.res, Reality=self.reality, Method="MWSS")
        return f


# form dictionary of all functions
function_dict = {**kernel_dict, **map_dict}
