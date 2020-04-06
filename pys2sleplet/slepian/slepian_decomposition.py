from dataclasses import dataclass

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.slepian_methods import choose_slepian_method


@dataclass
class SlepianDecomposition:
    L: int
    function: Functions
    flm: np.ndarray
    f: np.ndarray
    lambdas: np.ndarray
    s: np.ndarray
    thetas: np.ndarray
    delta_phi: float
    delta_theta: float

    def __post_init__(self) -> None:
        slepian = choose_slepian_method(self.L)
        self.lambdas = slepian.slepian_evals
        self.s = slepian.slepian_evecs
        self.thetas, phis = ssht.sample_positions(self.L, Method="MWSS")
        self.delta_phi = np.ediff1d(phis[0]).mean()
        self.delta_theta = np.ediff1d(self.thetas[:, 0]).mean()

    def decompose(self, rank: int, method: str = "harmonic_sum") -> np.ndarray:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        if method == "integrate_region":
            f_p = self._integrate_region(rank)
        elif method == "forward_transform":
            f_p = self._forward_transform(rank)
        elif method == "harmonic_sum":
            f_p = self._harmonic_sum(rank)
        else:
            raise ValueError(
                f"{method} is not a recognised Slepian decomposition method"
            )
        return f_p

    def _integrate_region(self, rank: int):
        """
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integrand = self.f * self.s[rank].conj()

        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        f_p = (integrand * weight).sum() / self.lambdas[rank]
        return f_p

    def _forward_transform(self, rank: int):
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        integrand = self.f * self.s[rank].conj()

        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        f_p = (integrand * weight).sum()
        return f_p

    def _harmonic_sum(self, rank: int):
        """
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m}
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        Y_{\ell m}(\omega) \overline{S_{p}(\omega)}
        """
        s_p_lm = ssht.forward(self.s[rank], self.L, Method="MWSS")

        summation = self.flm * s_p_lm.conj()

        # same as double sum with index
        f_p = summation.sum()
        return f_p
