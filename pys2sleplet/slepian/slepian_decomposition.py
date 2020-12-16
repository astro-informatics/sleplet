from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_region_sphere,
    integrate_whole_sphere,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import SAMPLING_SCHEME


@dataclass
class SlepianDecomposition:
    L: int
    slepian: SlepianFunctions
    f: np.ndarray
    flm: np.ndarray
    mask: np.ndarray
    _f: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _flm: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _mask: np.ndarray = field(default=None, init=False, repr=False)
    _slepian: SlepianFunctions = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._detect_method()

    def decompose(self, rank: int) -> complex:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        self._validate_rank(rank)

        if self.method == "harmonic_sum":
            return self._harmonic_sum(rank)
        elif self.method == "integrate_sphere":
            return self._integrate_sphere(rank)
        elif self.method == "integrate_region":
            return self._integrate_region(rank)
        else:
            raise ValueError(f"'{self.method}' is not a valid method")

    def decompose_all(self) -> np.ndarray:
        """
        decompose all ranks of the Slepian coefficients
        """
        coefficients = np.zeros(self.slepian.N, dtype=np.complex_)
        for rank in range(self.slepian.N):
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
        integration = integrate_region_sphere(self.f, s_p.conj(), weight, self.mask)
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
        return integrate_whole_sphere(self.f, s_p.conj(), weight)

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
        if rank >= self.slepian.N:
            raise ValueError(f"rank should be less than {self.slepian.N}")

    @property  # type:ignore
    def f(self) -> Optional[np.ndarray]:
        return self._f

    @f.setter
    def f(self, f: Optional[np.ndarray]) -> None:
        if isinstance(f, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            f = SlepianDecomposition._f
        self._f = f

    @property  # type:ignore
    def flm(self) -> Optional[np.ndarray]:
        return self._flm

    @flm.setter
    def flm(self, flm: Optional[np.ndarray]) -> None:
        if isinstance(flm, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            flm = SlepianDecomposition._flm
        self._flm = flm

    @property  # type:ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @property  # type: ignore
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @mask.setter
    def mask(self, mask: Optional[np.ndarray]) -> None:
        if isinstance(mask, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mask = SlepianDecomposition._mask
        self._mask = mask

    @property  # type:ignore
    def slepian(self) -> int:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian
