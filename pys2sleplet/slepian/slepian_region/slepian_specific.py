from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..slepian_functions import SlepianFunctions


class SlepianSpecific(SlepianFunctions):
    def __init__(
        self, L: int, phi_min: int, phi_max: int, theta_min: int, theta_max: int
    ) -> None:
        # self.matrix_name = Slepian.matrix_name + filename_region()
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        super().__init__(L)

    @abstractmethod
    def _create_annotations(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def phi_min(self) -> int:
        return self.__phi_min

    @phi_min.setter
    def phi_min(self, var: int) -> None:
        if var > self.phi_max:
            raise ValueError("phi_min greater than phi_max")
        elif var == self.phi_max:
            raise ValueError("phi_min same as phi_max")
        self.__phi_min = var

    @property
    def phi_max(self) -> int:
        return self.__phi_max

    @phi_max.setter
    def phi_max(self, var: int) -> None:
        if var < self.phi_min:
            raise ValueError("phi_max less than phi_min")
        elif var == self.phi_min:
            raise ValueError("phi_max same as phi_min")
        self.__phi_max = var

    @property
    def theta_min(self) -> int:
        return self.__theta_min

    @theta_min.setter
    def theta_min(self, var: int) -> None:
        if var > self.theta_max:
            raise ValueError("theta_min greater than theta_max")
        elif var == self.theta_max:
            raise ValueError("theta_min same as theta_max")
        self.__theta_min = var

    @property
    def theta_max(self) -> int:
        return self.__theta_max

    @theta_max.setter
    def theta_max(self, var: int) -> None:
        if var < self.theta_min:
            raise ValueError("theta_max less than theta_min")
        elif var == self.theta_min:
            raise ValueError("theta_max same as theta_min")
        self.__theta_max = var
