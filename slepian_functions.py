import sys
import os
import numpy as np
from scipy import integrate

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(self, L, theta_min=0, theta_max=np.pi, phi_min=0, phi_max=2 * np.pi):
        self.L = L
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.side = np.array(
            [ssht.elm2ind(ell, m) for ell in range(self.L) for m in range(ell + 1)]
        )

    def f(self, theta, phi, i, j):
        ylm = ssht.create_ylm(theta, phi, self.L)
        ylm = ylm.reshape(ylm.size)
        f = ylm[i] * np.conj(ylm[j]) * np.sin(theta)
        return f

    def real_func(self, theta, phi, i, j):
        return self.f(theta, phi, i, j).real

    def imag_func(self, theta, phi, i, j):
        return self.f(theta, phi, i, j).imag

    def integral(self, f, i, j):
        F = integrate.dblquad(
            f,
            self.phi_min,
            self.phi_max,
            lambda t: self.theta_min,
            lambda t: self.theta_max,
            args=(i, j),
        )[0]
        return F

    def D_integral(self, i, j):
        F_real = self.integral(self.real_func, i, j)
        F_imag = self.integral(self.imag_func, i, j)
        return F_real + 1j * F_imag

    def D_matrix(self):
        D = np.zeros((self.side.size, self.side.size), dtype=complex)
        for i in range(self.side.size):
            for j in range(i, self.side.size):
                integral = self.D_integral(self.side[i], self.side[j])
                D[i][j] = integral
                if i != j:
                    D[j][i] = np.conj(integral)
        return D


if __name__ == "__main__":
    L, theta0 = 2, np.pi / 9
    sf = SlepianFunctions(L, theta_max=theta0)
    D = sf.D_matrix()
    print(D)
