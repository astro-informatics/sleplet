from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.vars import SAMPLING_SCHEME


@dataclass
class SlepianDecomposition:
    function: Functions
    slepian: SlepianFunctions
    _L: int = field(init=False, repr=False)
    _flm: np.ndarray = field(init=False, repr=False)
    _f: np.ndarray = field(init=False, repr=False)
    _lambdas: np.ndarray = field(init=False, repr=False)
    _s: np.ndarray = field(init=False, repr=False)
    _theta_grid: np.ndarray = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_validation()
        self._L = self.function.L
        self._flm = self.function.multipole
        self._f = np.where(self.slepian.mask, self.function.field, 0)
        self._lambdas = self.slepian.eigenvalues
        self._s = self.slepian.eigenvectors
        self._theta_grid, phi_grid = ssht.sample_positions(
            self._L, Grid=True, Method=SAMPLING_SCHEME
        )
        delta_theta = np.ediff1d(self._theta_grid[:, 0]).mean()
        delta_phi = np.ediff1d(phi_grid[0]).mean()
        self._weight = self._theta_grid * delta_theta * delta_phi

    def decompose(self, rank: int, method: str) -> np.ndarray:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        if method == "integrate_region":
            f_p = self._integrate_region(rank)
        elif method == "integrate_sphere":
            f_p = self._integrate_sphere(rank)
        elif method == "forward_transform":
            f_p = self._forward_transform(rank)
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
        region = self.slepian.mask
        integrand = self._f * self._s[rank].conj()
        integral = (integrand * self._weight)[region].sum()
        f_p = integral / self._lambdas[rank]
        return f_p

    def _integrate_sphere(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integrand = self._f * self._s[rank].conj()
        f_p = (integrand * self._weight).sum()
        return f_p

    def _forward_transform(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m}
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        Y_{\ell m}(\omega) \overline{S_{p}(\omega)}
        """
        s_p_lm = ssht.forward(self._s[rank], self._L, Method=SAMPLING_SCHEME)
        summation = self._flm * s_p_lm.conj()

        # equivalent to looping through l and m
        f_p = summation.sum()
        return f_p

    def _input_validation(self) -> None:
        """
        check the bandlimits of the inputs agree
        """
        if self.function.L != self.slepian.L:
            raise AttributeError(
                f"bandlimits must agree: the function has an  of {self.function.L} "
                f"whereas the Slepian function has a L of {self.slepian.L}"
            )
