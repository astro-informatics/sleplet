import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.slepian_methods import choose_slepian_method


class SlepianDecomposition:
    def __init__(self, L: int, function: Functions) -> None:
        self.L = L
        slepian = choose_slepian_method(self.L)
        self.flm = function.multipole
        self.f = function.field
        self.lambdas = slepian.slepian_evals
        self.s = slepian.slepian_evecs

    def decompose(self, rank: int, method: str = "harmonic_sum") -> np.ndarray:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        if method == "integrate_region":
            f_p = self._integrate_region()
        elif method == "forward_transform":
            f_p = self._forward_transform()
        elif method == "harmonic_sum":
            f_p = self._harmonic_sum()
        else:
            raise ValueError(
                f"{method} is not a recognised Slepian decomposition method"
            )
        return f_p

    def _integrate_region(self):
        """
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        # function in integral
        function = self.signal * np.conj(self.s_p)

        # Jacobian
        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        # Slepian coefficient
        f_p = np.sum(function * weight) / self.lambda_p
        return f_p

    def _forward_transform(self):
        """
        f_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}
        """
        # function in integral
        function = self.f * np.conj(self.s_p)

        # Jacobian
        thetas, phis = ssht.sample_positions(self.L, Method="MWSS")
        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        # Slepian coefficient
        f_p = np.sum(function * weight)
        return f_p

    def _harmonic_sum(self):
        """
        f_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m}
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        Y_{\ell m}(\omega) \overline{S_{p}(\omega)}
        """
        # Slepian functions in harmonic space
        s_p_lm = ssht.forward(self.s_p, self.L, Method="MWSS")

        # Slepian coefficient
        f_p = s_p_lm.sum()
        return f_p
