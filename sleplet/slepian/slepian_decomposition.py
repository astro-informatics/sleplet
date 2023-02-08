from dataclasses import KW_ONLY

import numpy as np
import pyssht as ssht
from pydantic.dataclasses import dataclass

from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_region_sphere,
    integrate_whole_sphere,
)
from sleplet.utils.logger import logger
from sleplet.utils.validation import Validation
from sleplet.utils.vars import SAMPLING_SCHEME


@dataclass(config=Validation)
class SlepianDecomposition:
    L: int
    slepian: SlepianFunctions
    _: KW_ONLY
    f: np.ndarray | None = None
    flm: np.ndarray | None = None
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._detect_method()

    def decompose(self, rank: int) -> complex:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        self._validate_rank(rank)

        match self.method:  # noqa: E999
            case "harmonic_sum":
                return self._harmonic_sum(rank)
            case "integrate_sphere":
                return self._integrate_sphere(rank)
            case "integrate_region":
                return self._integrate_region(rank)
            case _:
                raise ValueError(f"'{self.method}' is not a valid method")

    def decompose_all(self, n_coefficients: int) -> np.ndarray:
        """
        decompose all ranks of the Slepian coefficients
        """
        coefficients = np.zeros(n_coefficients, dtype=np.complex_)
        for rank in range(n_coefficients):
            coefficients[rank] = self.decompose(rank)
        return coefficients

    def _integrate_region(self, rank: int) -> complex:
        r"""
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        s_p = ssht.inverse(
            self.slepian.eigenvectors[rank], self.L, Method=SAMPLING_SCHEME
        )
        weight = calc_integration_weight(self.L)
        assert isinstance(self.mask, np.ndarray)
        assert isinstance(self.f, np.ndarray)
        integration = integrate_region_sphere(self.mask, weight, self.f, s_p.conj())
        return integration / self.slepian.eigenvalues[rank]

    def _integrate_sphere(self, rank: int) -> complex:
        r"""
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        s_p = ssht.inverse(
            self.slepian.eigenvectors[rank], self.L, Method=SAMPLING_SCHEME
        )
        weight = calc_integration_weight(self.L)
        assert isinstance(self.f, np.ndarray)
        return integrate_whole_sphere(weight, self.f, s_p.conj())

    def _harmonic_sum(self, rank: int) -> complex:
        r"""
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}
        """
        return (self.flm * self.slepian.eigenvectors[rank].conj()).sum()

    def _detect_method(self) -> None:
        """
        detects what method is used to perform the decomposition
        """
        if isinstance(self.flm, np.ndarray):
            logger.info("harmonic sum method selected")
            self.method = "harmonic_sum"
        elif isinstance(self.f, np.ndarray) and not isinstance(self.mask, np.ndarray):
            logger.info("integrating the whole sphere method selected")
            self.method = "integrate_sphere"
        elif isinstance(self.f, np.ndarray):
            logger.info("integrating a region on the sphere method selected")
            self.method = "integrate_region"
        else:
            raise RuntimeError(
                "need to pass one off harmonic coefficients, real pixels "
                "or real pixels with a mask"
            )

    def _validate_rank(self, rank: int) -> None:
        """
        checks the requested rank is valid
        """
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.L**2:
            raise ValueError(f"rank should be less than {self.L**2}")
