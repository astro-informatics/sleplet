from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.logger import logger


@dataclass  # type: ignore
class Wavelet(Functions):
    _wavelets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        logger.info("start computing wavelets")
        wavelets = self._create_wavelets()
        logger.info("finish computing wavelets")
        self.multipole = wavelets[0] if self.j is None else wavelets[self.j + 1]

    @property  # type:ignore
    def wavelets(self) -> np.ndarray:
        return self._wavelets

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets

    @abstractmethod
    def _create_wavelets(self) -> np.ndarray:
        """
        method to create all the wavelets of the given type
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_reality(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_spin(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
