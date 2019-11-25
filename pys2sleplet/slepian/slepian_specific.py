from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.string_methods import filename_region


class SlepianSpecific(SlepianFunctions):
    def __init__(
        self, L: int, phi_min: int, phi_max: int, theta_min: int, theta_max: int
    ) -> None:
        super().__init__(L)
        self.matrix_filename = Path(
            SlepianFunctions.matrix_filename.name
            + filename_region(phi_min, phi_max, theta_min, theta_max)
        )
        self.is_polar_cap = (
            phi_min == 0 and phi_max == 360 and theta_min == 0 and theta_max != 180
        )
        self.phi_min = np.deg2rad(phi_min)
        self.phi_max = np.deg2rad(phi_max)
        self.theta_min = np.deg2rad(theta_min)
        self.theta_max = np.deg2rad(theta_max)

    @abstractmethod
    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def annotations(self) -> List[dict]:
        raise NotImplementedError

    @property
    def phi_min(self) -> int:
        return self.__phi_min

    @phi_min.setter
    def phi_min(self, var: int) -> None:
        if not isinstance(var, int):
            raise ValueError("phi min should be an integer")
        self.__phi_max = np.deg2rad(var)

    @property
    def phi_max(self) -> int:
        return self.__phi_max

    @phi_max.setter
    def phi_max(self, var: int) -> None:
        if not isinstance(var, int):
            raise ValueError("phi max should be an integer")
        # check that min/max phi values are correct
        if var > self.phi_max:
            self.__phi_min, self.phi_max = np.deg2rad(self.phi_max), np.deg2rad(var)
        elif var == self.phi_max:
            raise ValueError("Invalid region.")
        else:
            self.__phi_min = np.deg2rad(var)

    @property
    def theta_min(self) -> int:
        return self.__theta_min

    @theta_min.setter
    def theta_min(self, var: int) -> None:
        if not isinstance(var, int):
            raise ValueError("theta min should be an integer")
        self.__theta_min = np.deg2rad(var)

    @property
    def theta_max(self) -> int:
        return self.__theta_max

    @theta_max.setter
    def theta_max(self, var: int) -> None:
        if not isinstance(var, int):
            raise ValueError("theta max should be an integer")
        # check that min/max theta values are correct
        if var > self.theta_max:
            self.__theta_min, self.theta_max = (
                np.deg2rad(self.theta_max),
                np.deg2rad(var),
            )
        elif var == self.theta_max:
            raise ValueError("Invalid region.")
        else:
            self.__theta_max = np.deg2rad(var)
