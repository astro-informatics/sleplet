from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.integration_methods import (
    integrate_region_sphere,
    integrate_whole_sphere,
)
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import choose_slepian_method


@dataclass
class SlepianDecomposition:
    function: Functions
    _flm: np.ndarray = field(init=False, repr=False)
    _function: Functions = field(init=False, repr=False)
    _lambdas: np.ndarray = field(init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _s_p_lms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._flm = self.function.multipole
        self._region = self.function.region
        slepian = choose_slepian_method(self.function.L, self._region)
        self._lambdas = slepian.eigenvalues
        self._s_p_lms = slepian.eigenvectors

    def decompose(self, rank: int, method: str = "summation") -> np.ndarray:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        self._validate_rank(rank)

        if method == "integrate_region":
            f_p = self._integrate_region(rank)
        elif method == "integrate_sphere":
            f_p = self._integrate_sphere(rank)
        elif method == "summation":
            f_p = self._summation(rank)
        else:
            raise ValueError(
                f"{method} is not a recognised Slepian decomposition method"
            )
        return f_p

    def _integrate_region(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integration = integrate_region_sphere(
            self._flm, self._s_p_lms[rank], self._region, glm_conj=True
        )
        f_p = integration / self._lambdas[rank]
        return f_p

    def _integrate_sphere(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        f_p = integrate_whole_sphere(self._flm, self._s_p_lms[rank], glm_conj=True)
        return f_p

    def _summation(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}
        """
        f_p = (self._flm * self._s_p_lms[rank].conj()).sum()
        return f_p

    def _validate_rank(self, rank: int) -> None:
        """
        test the validity of the rank
        """
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        limit = self._s_p_lms.shape[0]
        if rank > limit:
            raise ValueError(f"rank should be no more than {limit}")

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
    def lambdas(self) -> np.ndarray:
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambdas: np.ndarray) -> None:
        self._lambdas = lambdas

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        self._region = region

    @property
    def s_p_lms(self) -> np.ndarray:
        return self._s_p_lms

    @s_p_lms.setter
    def s_p_lms(self, s_p_lms: np.ndarray) -> None:
        self._s_p_lms = s_p_lms
