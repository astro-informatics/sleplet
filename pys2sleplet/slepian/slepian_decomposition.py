from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_sphere,
)


@dataclass
class SlepianDecomposition:
    L: int
    flm: np.ndarray
    slepian: SlepianFunctions
    _flm: np.ndarray = field(init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _lambdas: np.ndarray = field(init=False, repr=False)
    _mask: np.ndarray = field(init=False, repr=False)
    _s_p_lms: np.ndarray = field(init=False, repr=False)
    _slepian: SlepianFunctions = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lambdas = self.slepian.eigenvalues
        self.mask = self.slepian.mask
        self.N = self.slepian.N
        self.s_p_lms = self.slepian.eigenvectors
        self.weight = calc_integration_weight(self.L)

    def decompose(self, rank: int, method: str = "harmonic_sum") -> complex:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        self._validate_rank(rank)

        if method == "integrate_region":
            f_p = self._integrate_region(rank)
        elif method == "integrate_sphere":
            f_p = self._integrate_sphere(rank)
        elif method == "harmonic_sum":
            f_p = self._harmonic_sum(rank)
        else:
            raise ValueError(
                f"{method} is not a recognised Slepian decomposition method"
            )
        return f_p

    def decompose_all(self, method: str = "harmonic_sum") -> np.ndarray:
        """
        decompose all ranks of the Slepian coefficients for a given method
        """
        coefficients = np.zeros(self.slepian.N, dtype=np.complex128)
        for rank in range(self.slepian.N):
            coefficients[rank] = self.decompose(rank, method=method)
        return coefficients

    def _integrate_region(self, rank: int) -> complex:
        r"""
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integration = integrate_sphere(
            self.L,
            self.flm,
            self.s_p_lms[rank],
            self.weight,
            glm_conj=True,
            mask=self.mask,
        )
        return integration / self.lambdas[rank]

    def _integrate_sphere(self, rank: int) -> complex:
        r"""
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        return integrate_sphere(
            self.L, self.flm, self.s_p_lms[rank], self.weight, glm_conj=True
        )

    def _harmonic_sum(self, rank: int) -> complex:
        r"""
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}
        """
        return (self.flm * self.s_p_lms[rank].conj()).sum()

    def _validate_rank(self, rank: int) -> None:
        """
        checks the requested rank is valid
        """
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= len(self.s_p_lms):
            raise ValueError(f"rank should be less than {len(self.s_p_lms)}")

    @property  # type:ignore
    def flm(self) -> np.ndarray:
        return self._flm

    @flm.setter
    def flm(self, flm: np.ndarray) -> None:
        self._flm = flm

    @property  # type:ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        self._N = N

    @property
    def lambdas(self) -> np.ndarray:
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambdas: np.ndarray) -> None:
        self._lambdas = lambdas

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    @property  # type:ignore
    def slepian(self) -> int:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian

    @property
    def s_p_lms(self) -> np.ndarray:
        return self._s_p_lms

    @s_p_lms.setter
    def s_p_lms(self, s_p_lms: np.ndarray) -> None:
        self._s_p_lms = s_p_lms

    @property
    def weight(self) -> np.ndarray:
        return self._weight

    @weight.setter
    def weight(self, weight: np.ndarray) -> None:
        self._weight = weight
