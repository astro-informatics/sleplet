from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
from slepian.slepian_functions import SlepianFunctions
from utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


class SlepianSpecific(SlepianFunctions):
    def __init__(
        self, L: int, phi_min: float, phi_max: float, theta_min: float, theta_max: float
    ) -> None:
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        super().__init__(L)

    @property
    def phi_min(self) -> float:
        return self.__phi_min

    @phi_min.setter
    def phi_min(self, var: float) -> None:
        if np.rad2deg(var) < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        elif np.rad2deg(var) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_min cannot be greater than {PHI_MAX_DEFAULT}")
        self.__phi_min = var

    @property
    def phi_max(self) -> float:
        return self.__phi_max

    @phi_max.setter
    def phi_max(self, var: float) -> None:
        if np.rad2deg(var) < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        elif np.rad2deg(var) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_max cannot be greater than {PHI_MAX_DEFAULT}")
        self.__phi_max = var

    @property
    def theta_min(self) -> float:
        return self.__theta_min

    @theta_min.setter
    def theta_min(self, var: float) -> None:
        if np.rad2deg(var) < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        elif np.rad2deg(var) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_min cannot be greater than {THETA_MAX_DEFAULT}")
        self.__theta_min = var

    @property
    def theta_max(self) -> float:
        return self.__theta_max

    @theta_max.setter
    def theta_max(self, var: float) -> None:
        if np.rad2deg(var) < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        elif np.rad2deg(var) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_max cannot be greater than {THETA_MAX_DEFAULT}")
        self.__theta_max = var

    @abstractmethod
    def _create_annotations(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def _create_fn_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
