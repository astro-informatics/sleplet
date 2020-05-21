from dataclasses import dataclass, field

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.vars import SAMPLING_SCHEME


@dataclass
class SlepianDecomposition:
    L: int
    function: Functions
    slepian: SlepianFunctions
    flm: np.ndarray = field(init=False, repr=False)
    f: np.ndarray = field(init=False, repr=False)
    lambdas: np.ndarray = field(init=False, repr=False)
    s: np.ndarray = field(init=False, repr=False)
    theta_grid: np.ndarray = field(init=False, repr=False)
    weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.flm = self.function.multipole
        self.f = np.where(self.slepian.mask, self.function.field, 0)
        self.lambdas = self.slepian.eigenvalues
        self.s = self.slepian.eigenvectors
        self.theta_grid, phi_grid = ssht.sample_positions(
            self.L, Grid=True, Method=SAMPLING_SCHEME
        )
        delta_theta = np.ediff1d(self.theta_grid[:, 0]).mean()
        delta_phi = np.ediff1d(phi_grid[0]).mean()
        self.weight = self.theta_grid * delta_theta * delta_phi

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
        integrand = self.f * self.s[rank].conj()
        integral = (integrand * self.weight)[region].sum()
        f_p = integral / self.lambdas[rank]
        return f_p

    def _integrate_sphere(self, rank: int) -> np.ndarray:
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integrand = self.f * self.s[rank].conj()
        f_p = (integrand * self.weight).sum()
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
        s_p_lm = ssht.forward(self.s[rank], self.L, Method=SAMPLING_SCHEME)
        summation = self.flm * s_p_lm.conj()

        # equivalent to looping through l and m
        f_p = summation.sum()
        return f_p
