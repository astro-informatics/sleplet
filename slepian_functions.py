from helper import Plotting
import numpy as np
import os
import quadpy
import sys
from typing import Tuple

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(
        self,
        L,
        phi_min: int = 0,
        phi_max: int = 360,
        theta_min: int = 0,
        theta_max: int = 180,
    ):
        self.L = L
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.N = self.L * self.L
        self.phi_max = phi_max
        self.phi_min = phi_min
        self.quad = quadpy.quadrilateral.rectangle_points(
            [np.deg2rad(theta_min), np.deg2rad(theta_max)],
            [np.deg2rad(phi_min), np.deg2rad(phi_max)],
        )
        self.scheme = quadpy.quadrilateral.cools_haegemans_1985_2()
        self.theta_max = theta_max
        self.theta_min = theta_min

    def f(self, omega: Tuple[complex, complex]) -> np.ndarray:
        theta, phi = omega
        ylm = ssht.create_ylm(theta, phi, self.L)
        ylmi, ylmj = ylm[self.i].reshape(-1), ylm[self.j].reshape(-1)
        f = ylmi * np.conj(ylmj) * np.sin(theta)
        return f

    def D_integral(self, i: int, j: int) -> complex:
        self.i, self.j = i, j
        F = self.scheme.integrate(self.f, self.quad)
        return F

    def D_matrix(self) -> np.ndarray:
        D = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            integral = self.D_integral(i, i)
            D[i][i] = integral
            for j in range(i + 1, self.N):
                integral = self.D_integral(i, j)
                D[i][j] = integral
                D[j][i] = np.conj(integral)
        return D

    def eigen_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        # numpy binary filename
        filename = os.path.join(
            self.location,
            "npy",
            (
                f"D_L-{self.L}_"
                f"pl-{self.phi_min}_pu-{self.phi_max}_"
                f"tl-{self.theta_min}_tu-{self.theta_max}.npy"
            ),
        )

        # check if file of D matrix already exists
        if os.path.exists(filename):
            D = np.load(filename)
        else:
            D = self.D_matrix()
            # save to speed up for future
            np.save(filename, D)

        eigen_values, eigen_vectors = np.linalg.eigh(D)
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        return eigen_values, eigen_vectors


if __name__ == "__main__":
    plotting = Plotting()
    L, theta0 = 4, 40
    sf = SlepianFunctions(L, theta_max=theta0)
    w, v = sf.eigen_problem()
    resolution = plotting.calc_resolution(L)
    flm = plotting.resolution_boost(v[0], L, resolution)
    f = ssht.inverse(flm, resolution)
    plotting.plotly_plot(f.real, "test")
