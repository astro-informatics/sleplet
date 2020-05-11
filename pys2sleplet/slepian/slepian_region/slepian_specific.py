from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


@dataclass  # type: ignore
class SlepianSpecific(SlepianFunctions):
    __order: int = field(init=False, repr=False)
    __phi_max: float = field(init=False, repr=False)
    __phi_min: float = field(init=False, repr=False)
    __theta_max: float = field(init=False, repr=False)
    __theta_min: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.order = 0
        self.phi_max = PHI_MIN_DEFAULT
        self.phi_min = PHI_MAX_DEFAULT
        self.theta_max = THETA_MAX_DEFAULT
        self.theta_min = THETA_MIN_DEFAULT

    @property
    def phi_min(self) -> float:
        return self.__phi_min

    @phi_min.setter
    def phi_min(self, phi_min: float) -> None:
        if np.rad2deg(phi_min) < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        elif np.rad2deg(phi_min) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_min cannot be greater than {PHI_MAX_DEFAULT}")
        self.__phi_min = phi_min

    @property
    def phi_max(self) -> float:
        return self.__phi_max

    @phi_max.setter
    def phi_max(self, phi_max: float) -> None:
        if np.rad2deg(phi_max) < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        elif np.rad2deg(phi_max) > PHI_MAX_DEFAULT:
            raise ValueError(f"phi_max cannot be greater than {PHI_MAX_DEFAULT}")
        self.__phi_max = phi_max

    @property
    def theta_min(self) -> float:
        return self.__theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if np.rad2deg(theta_min) < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        elif np.rad2deg(theta_min) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_min cannot be greater than {THETA_MAX_DEFAULT}")
        self.__theta_min = theta_min

    @property
    def theta_max(self) -> float:
        return self.__theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if np.rad2deg(theta_max) < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        elif np.rad2deg(theta_max) > THETA_MAX_DEFAULT:
            raise ValueError(f"theta_max cannot be greater than {THETA_MAX_DEFAULT}")
        self.__theta_max = theta_max

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int) -> None:
        if not isinstance(order, int):
            raise TypeError("order should be an integer")
        # check order is in correct range
        if abs(order) >= self.L:
            raise ValueError(f"Order magnitude should be less than {self.L}")
        self.__order = order

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
