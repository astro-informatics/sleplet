# import numpy as np
# import pyssht as ssht

# from pys2sleplet.flm.functions import Functions
# from pys2sleplet.slepian.slepian_functions import SlepianFunctions


# class SlepianDecomposition:
#     def __init__(self, function: Functions, slepian: SlepianFunctions) -> None:
#         self.flm = flm
#         self.f = ssht.inverse(flm, L, Method="MWSS")
#         self.L = L
#         self.lambda_p = self.slepian_evals[rank]
#         self.s_p = self.slepian_evecs[rank]

#     def decompose(self, rank: int, method: str = "harmonic_sum") -> np.ndarray:
#         """
#         decompose the signal into its Slepian coefficients via the given method
#         """
#         if method == "integrate_region":
#             f_p = self._integrate_region()
#         elif method == "forward_transform":
#             f_p = self._forward_transform()
#         elif method == "harmonic_sum":
#             f_p = self._harmonic_sum()
#         else:
#             raise ValueError(
#                 f"{method} is not a recognised Slepian decomposition method"
#             )
#         return f_p

#     def _integrate_region(self):
#         """
#         f_{p} =
#         \frac{1}{\lambda_{p}}
#         \int\limits_{R} \dd{\Omega(\omega)}
#         f(\omega) \overline{S_{p}(\omega)}
#         """
#         # function in integral
#         function = self.signal * np.conj(self.s_p)

#         # Jacobian
#         weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

#         # Slepian coefficient
#         f_p = np.sum(function * weight) / self.lambda_p
#         return f_p

#     def _forward_transform(self):
#         """
#         f_{p} =
#         \int\limits_{S^{2}} \dd{\Omega(\omega)}
#         f(\omega) \overline{S_{p}(\omega)}
#         """
#         # function in integral
#         function = self.f * np.conj(self.s_p)

#         # Jacobian
#         thetas, phis = ssht.sample_positions()
#         weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

#         # Slepian coefficient
#         f_p = np.sum(function * weight)
#         return f_p

#     def _harmonic_sum(self):
#         """
#         f_{p} =
#         \sum\limits_{\ell=0}^{L^{2}}
#         \sum\limits_{m=-\ell}^{\ell}
#         f_{\ell m}
#         \int\limits_{S^{2}} \dd{\Omega(\omega)}
#         Y_{\ell m}(\omega) \overline{S_{p}(\omega)}
#         """
#         # Slepian functions in harmonic space
#         s_p_lm = ssht.forward(self.s_p, self.L, Method="MWSS")

#         # Slepian coefficient
#         f_p = s_p_lm.sum()
#         return f_p
