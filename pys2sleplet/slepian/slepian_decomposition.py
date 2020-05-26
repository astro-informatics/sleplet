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
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_validation()
        self._L = self.function.L
        self._flm = self.function.multipole
        self._f = np.where(self.slepian.mask, self.function.field, 0)
        self._lambdas = self.slepian.eigenvalues
        theta_grid, phi_grid = ssht.sample_positions(
            self._L, Grid=True, Method=SAMPLING_SCHEME
        )
        delta_theta = np.ediff1d(theta_grid[:, 0]).mean()
        delta_phi = np.ediff1d(phi_grid[0]).mean()
        self._weight = theta_grid * delta_theta * delta_phi

    def decompose(self, rank: int, method: str) -> np.ndarray:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        s_p = ssht.inverse(
            self.slepian.eigenvectors[rank], self._L, Method=SAMPLING_SCHEME
        )
        l_p = self._lambdas[rank]

        if method == "integrate_region":
            f_p = self._integrate_region(s_p, l_p)
        elif method == "integrate_sphere":
            f_p = self._integrate_sphere(s_p)
        elif method == "forward_transform":
            f_p = self._forward_transform(s_p)
        else:
            raise ValueError(
                f"{method} is not a recognised Slepian decomposition method"
            )
        return f_p

    def _integrate_region(self, s_p: np.ndarray, l_p: float) -> np.ndarray:
        """
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        region = self.slepian.mask
        integrand = self._f * s_p.conj()
        integral = (integrand * self._weight)[region].sum()
        f_p = integral / l_p
        return f_p

    def _integrate_sphere(self, s_p: np.ndarray) -> np.ndarray:
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integrand = self._f * s_p.conj()
        f_p = (integrand * self._weight).sum()
        return f_p

    def _forward_transform(self, s_p: np.ndarray) -> np.ndarray:
        """
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m}
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        Y_{\ell m}(\omega) \overline{S_{p}(\omega)}
        """
        s_p_lm = ssht.forward(s_p, self._L, Method=SAMPLING_SCHEME)
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
