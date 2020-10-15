from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.convolution_methods import sifting_convolution
from pys2sleplet.utils.noise import compute_snr, create_noise
from pys2sleplet.utils.smoothing import apply_gaussian_smoothing
from pys2sleplet.utils.string_methods import filename_angle

_file_location = Path(__file__).resolve()


@dataclass  # type:ignore
class F_LM(Coefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def rotate(self, alpha: float, beta: float, gamma: float = 0) -> np.ndarray:
        return ssht.rotate_flms(self.coefficients, alpha, beta, gamma, self.L)

    def translate(self, alpha: float, beta: float) -> np.ndarray:
        # numpy binary filename
        filename = (
            _file_location.parents[1]
            / "data"
            / "trans_dirac"
            / f"trans_dd_L{self.L}_{filename_angle(alpha/np.pi,beta/np.pi)}.npy"
        )

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        if filename.exists():
            glm = np.load(filename)
        else:
            glm = ssht.create_ylm(beta, alpha, self.L).conj()
            glm = glm.reshape(glm.size)

            # save to speed up for future
            if settings.SAVE_MATRICES:
                np.save(filename, glm)

        # convolve with flm
        if self.name == "dirac_delta":
            coefficients = glm
        else:
            coefficients = self.convolve(self.coefficients, glm)
        return coefficients

    def convolve(self, flm: np.ndarray, glm: np.ndarray) -> np.ndarray:
        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False
        return sifting_convolution(flm, glm)

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise:
            nlm = create_noise(self.L, self.coefficients, self.noise)
            compute_snr(self.L, self.coefficients, nlm)
            self.coefficients += nlm

    def _smooth_signal(self) -> None:
        """
        applies Gaussian smoothing to the signal
        """
        if self.smoothing:
            self.coefficients = apply_gaussian_smoothing(
                self.coefficients, self.L, self.smoothing
            )

    @abstractmethod
    def _create_annotations(self) -> None:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _create_flm(self) -> None:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the function
        """
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
