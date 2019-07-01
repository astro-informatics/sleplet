import sys
import os
import numpy as np
from scipy import integrate

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(self, L, theta_min=0, theta_max=np.pi, phi_min=0, phi_max=2 * np.pi):
        self.L = L
        theta_grid, phi_grid = ssht.sample_positions(self.L, Grid=True)
        self.ylm = ssht.create_ylm(theta_grid, phi_grid, L)
        self.thetas, self.phis = ssht.sample_positions(L)
        self.theta_range = np.where(
            np.logical_and(self.thetas >= theta_min, self.thetas <= theta_max)
        )[0]
        self.phi_range = np.where(
            np.logical_and(self.phis >= phi_min, self.phis <= phi_max)
        )[0]
        self.side = np.array(
            [ssht.elm2ind(ell, m) for ell in range(self.L) for m in range(ell + 1)]
        )

    def f(self, i, j, theta_idx, phi_idx):
        return (
            self.ylm[i][theta_idx][phi_idx]
            * np.conj(self.ylm[j][theta_idx][phi_idx])
            * np.sin(self.thetas[theta_idx])
        )

    def D_integral(self, i, j):
        I = np.zeros(self.theta_range.size, dtype=complex)
        for t in self.theta_range:
            I[t] = integrate.simps(
                self.f(i, j, t, self.phi_range), self.phis[self.phi_range]
            )
        F = integrate.simps(I, self.thetas[self.theta_range])
        return F

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
    L = 16
    sf = SlepianFunctions(L)
    D = sf.D_matrix()
    print(D)
