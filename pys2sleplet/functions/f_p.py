from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pys2sleplet.functions.coefficients import Coefficients

_file_location = Path(__file__).resolve()


@dataclass  # type:ignore
class F_P(Coefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        pass

    def translate(self, alpha: float, beta: float) -> np.ndarray:
        pass

    def convolve(self, flm: np.ndarray, glm: np.ndarray) -> np.ndarray:
        pass

    def _add_noise_to_signal(self) -> None:
        pass

    def _smooth_signal(self) -> None:
        pass

    @abstractmethod
    def _create_annotations(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_flm(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
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
