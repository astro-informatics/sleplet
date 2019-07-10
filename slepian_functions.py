from plotting import Plotting
from memoization import cached
import numpy as np
import os
from scipy import integrate
import sys
from typing import List, Tuple

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(
        self, phi_min: int, phi_max: int, theta_min: int, theta_max: int, config: dict
    ):
        self.auto_open = config["auto_open"]
        self.L = config["L"]
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.N = self.L * self.L
        self.phi_max = phi_max
        self.phi_max_r = np.deg2rad(phi_max)
        self.phi_min = phi_min
        self.phi_min_r = np.deg2rad(phi_min)
        self.plotting = Plotting(
            auto_open=config["auto_open"], save_fig=config["save_fig"]
        )
        self.resolution = self.plotting.calc_resolution(config["L"])
        self.save_fig = config["save_fig"]
        self.theta_max = theta_max
        self.theta_max_r = np.deg2rad(theta_max)
        self.theta_min = theta_min
        self.theta_min_r = np.deg2rad(theta_min)
        self.plotting.missing_key(config, "annotation", True)
        self.plotting.missing_key(config, "type", None)
        self.eigen_values, self.eigen_vectors = self.eigen_problem()

    @cached
    def f(self, theta: float, phi: float, i: int, j: int) -> np.ndarray:
        ylm = ssht.create_ylm(theta, phi, self.L)
        ylmi, ylmj = ylm[i].reshape(-1), ylm[j].reshape(-1)
        f = ylmi * np.conj(ylmj) * np.sin(theta)
        return f

    def real_func(self, theta, phi, i, j):
        return self.f(theta, phi, i, j).real

    def imag_func(self, theta, phi, i, j):
        return self.f(theta, phi, i, j).imag

    def integral(self, f, i, j):
        F = integrate.dblquad(
            f,
            self.phi_min_r,
            self.phi_max_r,
            lambda t: self.theta_min_r,
            lambda t: self.theta_max_r,
            args=(i, j),
        )[0]
        return F

    def D_integral(self, i: int, j: int) -> complex:
        F_real = self.integral(self.real_func, i, j)
        F_imag = self.integral(self.imag_func, i, j)
        return F_real + 1j * F_imag

    def D_matrix(self) -> np.ndarray:
        D = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            # fill in diagonal components
            D[i][i] = self.D_integral(i, i)
            _, m_i = ssht.ind2elm(i)
            for j in range(i + 1, self.N):
                ell_j, m_j = ssht.ind2elm(j)
                # if possible to use previous calculations
                if m_i == 0 and m_j != 0 and ell_j < self.L:
                    # if positive m then use conjugate relation
                    if m_j > 0:
                        D[i][j] = self.D_integral(i, j)
                        D[j][i] = np.conj(D[i][j])
                        k = ssht.elm2ind(ell_j, -m_j)
                        D[i][k] = (-1) ** m_j * np.conj(D[i][j])
                        D[k][i] = np.conj(D[i][k])
                else:
                    integral = self.D_integral(i, j)
                    D[i][j] = integral
                    D[j][i] = np.conj(integral)
        return D

    def eigen_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        # numpy binary filename
        filename = os.path.join(
            self.location,
            "npy",
            "d_matrix",
            (f"D_L-{self.L}{self.filename_angle()}.npy"),
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
        eigen_values = eigen_values[idx] / np.max(eigen_values)
        eigen_vectors = eigen_vectors[:, idx]
        return eigen_values, eigen_vectors

    def plot(self, rank: int) -> None:
        """
        master plotting method
        """
        # setup
        print(f"Eigenvalue {rank}: {self.eigen_values[rank - 1]:.3f}")
        flm = self.eigen_vectors[rank - 1]
        filename = (
            f"slepian-{rank}_L-{self.L}{self.filename_angle()}_res-{self.resolution}_"
        )

        # boost resolution
        if self.resolution != self.L:
            flm = self.plotting.resolution_boost(flm, self.L, self.resolution)

        # inverse & plot
        f = ssht.inverse(flm, self.resolution)

        # check for plotting type
        if self.plotting.type == "real":
            f = f.real
        elif self.plotting.type == "imag":
            f = f.imag
        elif self.plotting.type == "abs":
            f = abs(f)
        elif self.plotting.type == "sum":
            f = f.real + f.imag

        # do plot
        filename += self.plotting.type
        self.plotting.plotly_plot(f, filename, self.annotations(), colourscheme="oxy")

    def filename_angle(self) -> str:
        """
        middle part of filename
        """
        # initialise filename
        filename = ""

        # if phi min is default
        if self.phi_min != 0:
            filename += f"_pmin-{self.phi_min}"

        # if phi max is default
        if self.phi_max != 360:
            filename += f"_pmax-{self.phi_max_}"

        # if theta min is default
        if self.theta_min != 0:
            filename += f"_tmin-{self.theta_min}"

        # if theta max is default
        if self.theta_max != 180:
            filename += f"_tmax-{self.theta_max}"
        return filename

    def annotations(self) -> List[dict]:
        if self.plotting.annotation:
            annotation = []
            config = dict(arrowcolor="black", arrowhead=6, ax=5, ay=5)
            ndots = 12
            for i in range(ndots):
                phi = 2 * np.pi / ndots * (i + 1)
                x, y, z = ssht.s2_to_cart(np.array(self.theta_max_r), np.array(phi))
                annotation.append({**dict(x=x, y=y, z=z), **config})
        else:
            annotation = []
        return annotation
