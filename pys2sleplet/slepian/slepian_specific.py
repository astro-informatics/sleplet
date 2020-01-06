from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from pys2sleplet.flm.kernels.slepian import Slepian
from pys2sleplet.utils.string_methods import filename_region
from pys2sleplet.utils.vars import SLEPIAN


class SlepianSpecific(Slepian):
    def __init__(self, L: int) -> None:
        super().__init__(L)
        self.matrix_name = Slepian.matrix_name + filename_region()
        self.is_polar_cap = (
            SLEPIAN["PHI_MIN"] == 0
            and SLEPIAN["PHI_MAX"] == 360
            and SLEPIAN["THETA_MIN"] == 0
            and SLEPIAN["THETA_MAX"] != 180
        )
        self.phi_min = np.deg2rad(SLEPIAN["PHI_MIN"])
        self.phi_max = np.deg2rad(SLEPIAN["PHI_MAX"])
        self.theta_min = np.deg2rad(SLEPIAN["THETA_MIN"])
        self.theta_max = np.deg2rad(SLEPIAN["THETA_MAX"])

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
