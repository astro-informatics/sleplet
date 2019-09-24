import numpy as np
import pyssht as ssht
from typing import Tuple


class SlepianArbitrary:
    def __init__(
        self, L: int, phi_min: int, phi_max: int, theta_min: int, theta_max: int
    ) -> None:
        samples = self.calc_samples(L)
        theta, phi = ssht.sample_positions(samples, Method="MWSS")
        thetas, phis = ssht.sample_positions(samples, Grid=True, Method="MWSS")
        phi_mask = np.where(
            (phi >= np.deg2rad(phi_min)) & (phi <= np.deg2rad(phi_max))
        )[0]
        theta_mask = np.where(
            (theta >= np.deg2rad(theta_min)) & (theta <= np.deg2rad(theta_max))
        )[0]
        ylm = ssht.create_ylm(thetas, phis, L)
        self.delta_phi = np.mean(np.ediff1d(phi))
        self.delta_theta = np.mean(np.ediff1d(theta))
        self.L = L
        self.N = L * L
        self.thetas = thetas[theta_mask[:, np.newaxis], phi_mask]
        self.ylm = ylm[:, theta_mask[:, np.newaxis], phi_mask]

    def f(self, i: int, j: int) -> np.ndarray:
        f = self.ylm[i] * np.conj(self.ylm[j])
        return f

    def w(self) -> np.ndarray:
        w = np.sin(self.thetas) * self.delta_theta * self.delta_phi
        return w

    def integral(self, i: int, j: int) -> complex:
        F = np.sum(self.f(i, j) * self.w())
        return F

    def matrix(self) -> np.ndarray:
        # initialise
        D = np.zeros((self.N, self.N), dtype=complex)

        for i in range(self.N):
            # fill in diagonal components
            D[i][i] = self.integral(i, i)
            _, m_i = ssht.ind2elm(i)
            for j in range(i + 1, self.N):
                ell_j, m_j = ssht.ind2elm(j)
                # if possible to use previous calculations
                if m_i == 0 and m_j != 0 and ell_j < self.L:
                    # if positive m then use conjugate relation
                    if m_j > 0:
                        D[i][j] = self.integral(i, j)
                        D[j][i] = np.conj(D[i][j])
                        k = ssht.elm2ind(ell_j, -m_j)
                        D[i][k] = (-1) ** m_j * np.conj(D[i][j])
                        D[k][i] = np.conj(D[i][k])
                else:
                    D[i][j] = self.integral(i, j)
                    D[j][i] = np.conj(D[i][j])
        return D

    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        D = self.matrix()

        # solve eigenproblem
        eigenvalues, eigenvectors = np.linalg.eigh(D)

        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = np.conj(eigenvectors[:, idx]).T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors

    @staticmethod
    def calc_samples(L: int) -> int:
        """
        calculate appropriate sample number for given L
        chosen such that have a two samples less than 1deg
        """
        if L == 1:
            samples = 180
        elif L < 4:
            samples = 90
        elif L < 8:
            samples = 45
        elif L < 16:
            samples = 23
        elif L < 32:
            samples = 12
        elif L < 64:
            samples = 6
        elif L < 128:
            samples = 3
        elif L < 256:
            samples = 2
        else:
            samples = 1
        return samples
