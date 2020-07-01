from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.integration_methods import integrate_sphere
from pys2sleplet.utils.slepian_methods import choose_slepian_method
from pys2sleplet.utils.vars import DECOMPOSITION_DEFAULT


@dataclass
class SlepianDecomposition:
    function: Functions
    _L: int = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _flm: np.ndarray = field(init=False, repr=False)
    _function: Functions = field(init=False, repr=False)
    _lambdas: np.ndarray = field(init=False, repr=False)
    _mask: np.ndarray = field(init=False, repr=False)
    _resolution: int = field(init=False, repr=False)
    _s_p_lms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.L = self.function.L
        self.flm = self.function.multipole
        region = self.function.region
        slepian = choose_slepian_method(self.L, region)
        self.lambdas = slepian.eigenvalues
        self.mask = slepian.mask
        self.resolution = slepian.resolution
        self.s_p_lms = slepian.eigenvectors
        self.N = self.s_p_lms.shape[0]

    def decompose(self, rank: int, method: str = DECOMPOSITION_DEFAULT) -> complex:
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

    def decompose_all(self, method: str = DECOMPOSITION_DEFAULT) -> np.ndarray:
        """
        decompose all ranks of the Slepian coefficients for a given method
        """
        coefficients = np.zeros(self.N, dtype=complex)
        for rank in range(self.N):
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
            self.resolution,
            self.flm,
            self.s_p_lms[rank],
            glm_conj=True,
            mask_boosted=self.mask,
        )
        f_p = integration / self.lambdas[rank]
        return f_p

    def _integrate_sphere(self, rank: int) -> complex:
        r"""
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        f_p = integrate_sphere(
            self.L, self.resolution, self.flm, self.s_p_lms[rank], glm_conj=True
        )
        return f_p

    def _harmonic_sum(self, rank: int) -> complex:
        r"""
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}
        """
        f_p = (self.flm * self.s_p_lms[rank].conj()).sum()
        return f_p

    def _validate_rank(self, rank: int) -> None:
        """
        checks the requested rank is valid
        """
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.N:
            raise ValueError(f"rank should be less than {self.N}")

    @property
    def flm(self) -> np.ndarray:
        return self._flm

    @flm.setter
    def flm(self, flm: np.ndarray) -> None:
        self._flm = flm

    @property  # type: ignore
    def function(self) -> Functions:
        return self._function

    @function.setter
    def function(self, function: Functions) -> None:
        if function.region is None:
            raise AttributeError(
                f"{function.__class__.__name__} needs to have a region passed to it"
            )
        self._function = function

    @property
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

    @property
    def resolution(self) -> int:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int) -> None:
        self._resolution = resolution

    @property
    def s_p_lms(self) -> np.ndarray:
        return self._s_p_lms

    @s_p_lms.setter
    def s_p_lms(self, s_p_lms: np.ndarray) -> None:
        self._s_p_lms = s_p_lms
