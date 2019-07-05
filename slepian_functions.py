import sys
import os
import numpy as np
import quadpy

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
        self.scheme = quadpy.quadrilateral.cools_haegemans_1985_2()
        self.quad = quadpy.quadrilateral.rectangle_points(
            [self.theta_min, self.theta_max], [self.phi_min, self.phi_max]
        )

    def f(self, omega):
        theta, phi = omega
        ylm = ssht.create_ylm(theta, phi, self.L)
        ylmi, ylmj = ylm[self.i].reshape(-1), ylm[self.j].reshape(-1)
        f = ylmi * np.conj(ylmj) * np.sin(theta)
        return f

    def D_integral(self, i, j):
        self.i, self.j = i, j
        F = self.scheme.integrate(self.f, self.quad)
        return F

    def D_matrix(self):
        D = np.zeros((self.side.size, self.side.size), dtype=complex)
        for i in range(self.side.size):
            integral = self.D_integral(self.side[i], self.side[i])
            D[i][i] = integral
            for j in range(i + 1, self.side.size):
                integral = self.D_integral(self.side[i], self.side[j])
                D[i][j] = integral
                D[j][i] = np.conj(integral)
        return D


if __name__ == "__main__":
    L, theta0 = 4, np.pi / 9
    sf = SlepianFunctions(L, theta_max=theta0)
    D = sf.D_matrix()
    print(D)
