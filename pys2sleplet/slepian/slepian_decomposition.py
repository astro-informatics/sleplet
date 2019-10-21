# from helper import calc_samples
import numpy as np
import pyssht as ssht


class SlepianDecomposition:
    def __init__(self, flm, L, slepian_evals, slepian_evecs, rank):
        self.flm = flm
        self.f = ssht.inverse(flm, L, Method="MWSS")
        self.L = L
        self.lambda_p = self.slepian_evals[rank]
        self.s_p = self.slepian_evecs[rank]

    def method_one(self):
        # function in integral
        function = self.signal * np.conj(self.s_p)

        # Jacobian
        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        # Slepian coefficient
        f_p = np.sum(function * weight) / self.lambda_p
        return f_p

    def method_two(self):
        # function in integral
        function = self.f * np.conj(self.s_p)

        # Jacobian
        thetas, phis = ssht.sample_positions()
        weight = np.sin(self.thetas) * self.delta_theta * self.delta_phi

        # Slepian coefficient
        f_p = np.sum(function * weight)
        return f_p

    def method_three(self):
        # Slepian functions in harmonic space
        s_p_lm = ssht.forward(self.s_p, self.L, Method="MWSS")

        # Slepian coefficient
        f_p = s_p_lm.sum()
        return f_p
