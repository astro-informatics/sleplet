from plotting import Plotting
import numpy as np
import os
import quadpy
import sys

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(self, L, theta_min=0, theta_max=np.pi, phi_min=0, phi_max=2 * np.pi):
        self.L = L
        self.N = self.L * self.L
        self.scheme = quadpy.quadrilateral.cools_haegemans_1985_2()
        self.quad = quadpy.quadrilateral.rectangle_points(
            [theta_min, theta_max], [phi_min, phi_max]
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
        D = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            integral = self.D_integral(i, i)
            D[i][i] = integral
            for j in range(i + 1, self.N):
                integral = self.D_integral(i, j)
                D[i][j] = integral
                D[j][i] = np.conj(integral)
        return D

    def eigen_problem(self):
        D = self.D_matrix()
        e_funs, e_vals = np.linalg.eigh(D)
        return e_funs, e_vals


if __name__ == "__main__":
    plotting = Plotting()
    L, theta0 = 2, np.pi / 9
    sf = SlepianFunctions(L, theta_max=theta0)
    _, eigenfunctions = sf.eigen_problem()
    resolution = plotting.calc_resolution(L)
    for v in eigenfunctions:
        flm = plotting.resolution_boost(v, L, resolution)
        f = ssht.inverse(flm, resolution)
        plotting.plotly_plot(f.real, "test")
